"""506MB F32 모델의 텐서 이름을 현재 추론 엔진 형식으로 변환하여 re-export.

변환: layers.X.mixing.{fwd,bwd}.mamba2.YYY → layers.X.mixing.{fwd,bwd}.YYY
     (mamba2. prefix 제거)
"""
import struct, sys, os

def reexport(src_path, dst_path):
    with open(src_path, 'rb') as f:
        data = f.read()

    # parse header
    off = 0
    magic = data[off:off+4]; off += 4
    assert magic == b'BMMQ'
    version = struct.unpack_from('<H', data, off)[0]; off += 2
    n_tensors = struct.unpack_from('<I', data, off)[0]; off += 4

    # parse all tensors → collect (name, raw_chunk)
    tensors = []
    for _ in range(n_tensors):
        nl = struct.unpack_from('<H', data, off)[0]; off += 2
        name = data[off:off+nl].decode('utf-8'); off += nl
        dtype = data[off]; off += 1
        ndim = data[off]; off += 1
        shape = []
        for _ in range(ndim):
            shape.append(struct.unpack_from('<I', data, off)[0]); off += 4
        data_len = struct.unpack_from('<Q', data, off)[0]; off += 8
        raw = data[off:off+data_len]; off += data_len
        tensors.append((name, dtype, ndim, shape, raw))

    # rename
    renamed = []
    for name, dtype, ndim, shape, raw in tensors:
        new_name = name.replace('.mamba2.', '.')
        if new_name != name:
            print(f'  {name} → {new_name}')
        renamed.append((new_name, dtype, ndim, shape, raw))

    # write
    with open(dst_path, 'wb') as f:
        f.write(b'BMMQ')
        f.write(struct.pack('<H', version))
        f.write(struct.pack('<I', len(renamed)))
        for name, dtype, ndim, shape, raw in renamed:
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('<H', len(name_bytes)))
            f.write(name_bytes)
            f.write(struct.pack('B', dtype))
            f.write(struct.pack('B', ndim))
            for s in shape:
                f.write(struct.pack('<I', s))
            f.write(struct.pack('<Q', len(raw)))
            f.write(raw)

    print(f'\n{src_path} ({os.path.getsize(src_path)//1024//1024}MB) → {dst_path} ({os.path.getsize(dst_path)//1024//1024}MB)')

if __name__ == '__main__':
    reexport(
        'exp-2-pass-consensus/exported/model.bmmq',
        'exp-2-pass-consensus/exported/model_renamed.bmmq'
    )
