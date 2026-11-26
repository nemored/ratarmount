# pylint: disable=wrong-import-position
# pylint: disable=protected-access

"""Deferred mounting and the limit on open archives in AutoMountLayer."""

import contextlib
import io
import os
import random
import shutil
import stat
import sys
import tempfile
import zipfile

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from helpers import find_test_file

from ratarmountcore.mountsource.compositing.automount import AutoMountLayer
from ratarmountcore.mountsource.factory import open_mount_source
from ratarmountcore.utils import RatarmountError

ARCHIVES = ['a', 'b', 'c', 'd', 'e']


def zip_bytes(members: dict) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as archive:
        for name, data in members.items():
            archive.writestr(name, data)
    return buffer.getvalue()


def content(name: str) -> bytes:
    return f"contents of {name}\n".encode() * 2000


@contextlib.contextmanager
def archive_folder(**options):
    """A folder of archives behind an AutoMountLayer with lazy mounting."""
    with tempfile.TemporaryDirectory() as folder:
        for name in ARCHIVES:
            with open(os.path.join(folder, f"{name}.zip"), 'wb') as file:
                file.write(zip_bytes({'dir/file.txt': content(name), 'other.txt': content(name.upper())}))
        inner = zip_bytes({'deep.txt': content('deep')})
        with open(os.path.join(folder, 'nested.zip'), 'wb') as file:
            file.write(zip_bytes({'inner.zip': inner, 'beside.txt': content('beside')}))
        with open(os.path.join(folder, 'bad.zip'), 'wb') as file:
            file.write(b'this is not an archive')
        with open(os.path.join(folder, 'plain.txt'), 'wb') as file:
            file.write(content('plain'))

        options = {'recursive': True, 'lazyMounting': True, 'recursiveExtensions': ['.zip']} | options
        with open_mount_source(folder, **options) as source, AutoMountLayer(source, **options) as layer:
            yield folder, layer


def mounted(layer) -> set:
    return set(layer.mounted) - {'/'}


def read(layer, path: str) -> bytes:
    fileInfo = layer.lookup(path)
    assert fileInfo is not None, path
    with layer.open(fileInfo) as file:
        return file.read()


def test_without_the_options_looking_up_an_archive_mounts_it():
    with archive_folder() as (_, layer):
        assert stat.S_ISREG(layer.list_mode('/')['a.zip'])
        assert stat.S_ISDIR(layer.lookup('/a.zip').mode)
        assert mounted(layer) == {'/a.zip'}


def test_listing_shows_folders_without_mounting():
    with archive_folder(deferMounting=True) as (_, layer):
        modes = layer.list_mode('/')
        infos = layer.list('/')
        for name in [f"{name}.zip" for name in ARCHIVES] + ['nested.zip']:
            assert stat.S_ISDIR(modes[name]), name
            assert stat.S_ISDIR(infos[name].mode), name
            fileInfo = layer.lookup('/' + name)
            assert stat.S_ISDIR(fileInfo.mode)
            assert fileInfo.mode & stat.S_IXUSR
            assert layer.versions('/' + name) == 1
            assert layer.list_xattr(fileInfo) == []
            assert layer.get_xattr(fileInfo, 'user.anything') is None
        assert stat.S_ISREG(modes['plain.txt'])
        assert layer.lookup('/missing.zip') is None
        assert not mounted(layer)


def test_looking_inside_mounts_only_that_archive():
    with archive_folder(deferMounting=True) as (_, layer):
        before = layer.lookup('/a.zip')
        assert set(layer.list('/a.zip')) == {'dir', 'other.txt'}
        assert mounted(layer) == {'/a.zip'}
        # The folder looks the same once mounted.
        after = layer.lookup('/a.zip')
        assert (before.mode, before.size, before.mtime) == (after.mode, after.size, after.mtime)
        assert stat.S_ISDIR(layer.list_mode('/')['a.zip'])
        assert read(layer, '/b.zip/dir/file.txt') == content('b')
        assert mounted(layer) == {'/a.zip', '/b.zip'}


def test_unreadable_archive_turns_back_into_a_file():
    with archive_folder(deferMounting=True) as (folder, layer):
        assert stat.S_ISDIR(layer.lookup('/bad.zip').mode)
        assert layer.list('/bad.zip') is None
        assert stat.S_ISREG(layer.lookup('/bad.zip').mode)
        assert stat.S_ISREG(layer.list_mode('/')['bad.zip'])
        assert read(layer, '/bad.zip') == b'this is not an archive'
        # A replaced file gets another chance.
        with open(os.path.join(folder, 'bad.zip'), 'wb') as file:
            file.write(zip_bytes({'now.txt': b'fine'}))
        assert stat.S_ISDIR(layer.lookup('/bad.zip').mode)
        assert read(layer, '/bad.zip/now.txt') == b'fine'


def test_nested_archives_are_deferred_too():
    with archive_folder(deferMounting=True) as (_, layer):
        assert stat.S_ISDIR(layer.list_mode('/nested.zip')['inner.zip'])
        assert mounted(layer) == {'/nested.zip'}
        assert read(layer, '/nested.zip/inner.zip/deep.txt') == content('deep')
        assert mounted(layer) == {'/nested.zip', '/nested.zip/inner.zip'}


def test_limit_unmounts_the_least_recently_used():
    with archive_folder(maxOpenArchives=2) as (_, layer):
        assert read(layer, '/a.zip/dir/file.txt') == content('a')
        assert read(layer, '/b.zip/dir/file.txt') == content('b')
        assert read(layer, '/a.zip/other.txt') == content('A')
        assert read(layer, '/c.zip/dir/file.txt') == content('c')
        assert mounted(layer) == {'/a.zip', '/c.zip'}
        assert read(layer, '/b.zip/dir/file.txt') == content('b')
        assert mounted(layer) == {'/b.zip', '/c.zip'}


def test_unmounting_closes_the_archive():
    with archive_folder(maxOpenArchives=1) as (_, layer):
        read(layer, '/a.zip/dir/file.txt')
        archive = layer.mounted['/a.zip'].mountSource.fileObject
        assert archive.fp is not None
        read(layer, '/b.zip/dir/file.txt')
        assert archive.fp is None


def test_file_info_outlives_unmounting():
    with archive_folder(maxOpenArchives=1, deferMounting=True) as (_, layer):
        fileInfo = layer.lookup('/a.zip/dir/file.txt')
        folderInfo = layer.lookup('/a.zip')
        read(layer, '/b.zip/dir/file.txt')
        assert mounted(layer) == {'/b.zip'}
        assert layer.list_xattr(folderInfo) == []
        assert mounted(layer) == {'/b.zip'}
        with layer.open(fileInfo) as file:
            assert file.read() == content('a')


def test_open_files_keep_their_archive_mounted():
    with archive_folder(maxOpenArchives=1) as (_, layer):
        first = layer.open(layer.lookup('/a.zip/dir/file.txt'))
        second = layer.open(layer.lookup('/a.zip/other.txt'))
        assert first.read(10) == content('a')[:10]
        for name in ['b', 'c', 'd']:
            assert read(layer, f'/{name}.zip/dir/file.txt') == content(name)
            assert '/a.zip' in mounted(layer)
        # Seeking in both directions still works after the churn.
        first.seek(1000)
        assert first.read(10) == content('a')[1000:1010]
        first.seek(5)
        assert first.read(10) == content('a')[5:15]
        first.close()
        first.close()
        read(layer, '/e.zip/dir/file.txt')
        assert '/a.zip' in mounted(layer), "the second file is still open"
        assert second.read() == content('A')
        second.close()
        assert len(mounted(layer)) == 1


def test_forgotten_files_release_their_archive():
    with archive_folder(maxOpenArchives=1) as (_, layer):
        file = layer.open(layer.lookup('/a.zip/dir/file.txt'))
        del file
        read(layer, '/b.zip/dir/file.txt')
        assert mounted(layer) == {'/b.zip'}


def test_limit_keeps_parents_of_nested_archives():
    with archive_folder(maxOpenArchives=1, deferMounting=True) as (_, layer):
        deep = layer.open(layer.lookup('/nested.zip/inner.zip/deep.txt'))
        assert mounted(layer) == {'/nested.zip', '/nested.zip/inner.zip'}
        read(layer, '/a.zip/dir/file.txt')
        assert {'/nested.zip', '/nested.zip/inner.zip'} <= mounted(layer)
        assert deep.read() == content('deep')
        archiveFile = layer.mounted['/nested.zip/inner.zip'].archiveFile
        deep.close()
        read(layer, '/b.zip/dir/file.txt')
        assert mounted(layer) == {'/b.zip'}
        assert archiveFile.closed
        assert read(layer, '/nested.zip/inner.zip/deep.txt') == content('deep')
        assert read(layer, '/nested.zip/beside.txt') == content('beside')


def test_random_access_stays_correct_and_bounded():
    expected = {f'/{name}.zip/dir/file.txt': content(name) for name in ARCHIVES}
    expected |= {f'/{name}.zip/other.txt': content(name.upper()) for name in ARCHIVES}
    expected['/nested.zip/inner.zip/deep.txt'] = content('deep')
    expected['/nested.zip/beside.txt'] = content('beside')
    expected['/plain.txt'] = content('plain')

    for limit in [1, 2, 3]:
        for defer in [False, True]:
            with archive_folder(maxOpenArchives=limit, deferMounting=defer) as (_, layer):
                generator = random.Random(limit * 2 + defer)
                openFiles = []
                for _ in range(600):
                    action = generator.random()
                    if action < 0.1:
                        layer.list('/')
                        layer.list_mode(f'/{generator.choice(ARCHIVES)}.zip')
                    elif action < 0.4 and openFiles:
                        file, data = generator.choice(openFiles)
                        offset = generator.randrange(len(data))
                        file.seek(offset)
                        assert file.read(100) == data[offset : offset + 100]
                    elif action < 0.55 and openFiles:
                        openFiles.pop(generator.randrange(len(openFiles)))[0].close()
                    else:
                        path = generator.choice(list(expected))
                        if len(openFiles) < 4:
                            openFiles.append((layer.open(layer.lookup(path)), expected[path]))
                        else:
                            assert read(layer, path) == expected[path]
                    # At most the limit, plus what open files and their parents hold.
                    assert len(mounted(layer)) <= limit + 2 * len(openFiles) + 1
                for file, _ in openFiles:
                    file.close()
                assert read(layer, '/a.zip/dir/file.txt') == content('a')
                assert len(mounted(layer)) <= limit
                assert all(info.openFiles == 0 for info in layer.mounted.values())


def all_files(layer, path: str = '/') -> dict:
    """Every regular file below a path, with its contents."""
    result = {}
    for name in layer.list_mode(path) or []:
        child = os.path.join(path, name)
        fileInfo = layer.lookup(child)
        if fileInfo is None:
            continue
        if stat.S_ISDIR(fileInfo.mode):
            result |= all_files(layer, child)
        elif stat.S_ISREG(fileInfo.mode):
            with layer.open(fileInfo) as file:
                result[child] = file.read()
    return result


def test_rar_archives_can_be_unmounted_and_mounted_again():
    try:
        import rarfile  # noqa: F401  # pylint: disable=import-outside-toplevel,unused-import
    except ImportError:
        return

    names = ['folder-symlink.rar', 'file-in-non-existing-folder.rar', 'denormal-paths.rar']
    with tempfile.TemporaryDirectory() as folder:
        for name in names:
            shutil.copy(find_test_file(name), os.path.join(folder, name))
        options = {'recursive': True, 'lazyMounting': True, 'recursiveExtensions': ['.rar']}
        with open_mount_source(folder, **options) as source, AutoMountLayer(source, **options) as layer:
            expected = all_files(layer)
        assert {path.split('/')[1] for path in expected} == set(names)

        options |= {'maxOpenArchives': 1, 'deferMounting': True}
        with open_mount_source(folder, **options) as source, AutoMountLayer(source, **options) as layer:
            assert all_files(layer) == expected
            paths = sorted(expected) * 4
            random.Random(0).shuffle(paths)
            for path in paths:
                assert read(layer, path) == expected[path]
                assert len(mounted(layer)) <= 1


def test_options_need_lazy_mounting_and_plain_mount_points():
    for options in [
        {'maxOpenArchives': 2, 'lazyMounting': False},
        {'deferMounting': True, 'lazyMounting': False},
        {'deferMounting': True, 'stripRecursiveTarExtension': True},
        {'maxOpenArchives': 2, 'transformRecursiveMountPoint': ('[.]zip$', '')},
    ]:
        try:
            with archive_folder(**options):
                pass
        except RatarmountError:
            continue
        raise AssertionError(f"accepted {options}")
