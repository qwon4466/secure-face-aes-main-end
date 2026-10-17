import numpy as np
from core.face_display import annotate,korean_font
from core.selective_crypto import protect,generate_keys,keyset,restore_frames
from core.selective_recording import Recorder

def test_status_color_and_original_integrity():
    frame=np.random.default_rng(7).integers(0,255,(320,480,3),dtype=np.uint8);before=frame.copy()
    faces=[{'bbox':[20,5,180,200],'group':'internal','authorized':True},
           {'bbox':[270,10,450,230],'group':'master','authorized':False}]
    protected,masks,records=protect(frame,faces)
    assert np.array_equal(frame,before)
    assert np.array_equal(protected[5,20],[0,255,0])
    assert np.array_equal(protected[10,270],[0,0,255])
    assert not masks['external'].any()
    # Strong smoothing in the central face area, excluding label/border.
    assert protected[70:150,60:130].std()<frame[70:150,60:130].std()/8
    assert korean_font().getmask('허가자').getbbox()

def test_single_group_pixels_stay_exclusive_for_selective_restore():
    frame=np.random.default_rng(17).integers(0,255,(240,320,3),dtype=np.uint8)
    box=[40,30,180,170]
    protected,masks,_=protect(frame,[{'bbox':box,'group':'internal','authorized':True}])
    assert masks['internal'].sum()==(box[2]-box[0])*(box[3]-box[1])
    assert not masks['external'].any() and not masks['master'].any()
    assert np.array_equal(protected[200,250],frame[200,250])

def test_encrypted_buffer_and_label_restore(work):
    keys_dir=work/'keys';generate_keys(keys_dir);keys=keyset(keys_dir)
    frame=np.random.default_rng(8).integers(0,255,(240,320,3),dtype=np.uint8)
    faces=[{'bbox':[50,20,250,210],'group':'master','authorized':False}]
    recorder=Recorder(work/'chunks',keys,chunk_frames=2)
    recorder.add(frame,faces,0,0)
    nonce,ciphertext=recorder.frames[0]
    assert ciphertext!=frame.tobytes() and len(ciphertext)==frame.nbytes+16
    recorder.finish()
    chunk=work/'chunks'/recorder.chunks[0]
    restored,_,_=restore_frames(chunk,keys['master'])
    assert np.array_equal(restored[0],frame)
    external,_,_=restore_frames(chunk,keys['external'])
    assert np.array_equal(external[0],protect(frame,faces)[0])
