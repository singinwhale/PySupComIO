import io


def read_null_terminated_string(offset: int, file_reader: io.BufferedReader):
    file_reader.seek(offset)
    return ''.join(iter(lambda: file_reader.read(1).decode('ascii'), '\x00'))
