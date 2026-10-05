import cv2, numpy as np
from fit import upright, D
from edit import srgb2lin, lin2srgb, wb, sharpen
def local_tone(lin, target, alpha, gmax):
    Y=0.0722*lin[...,0]+0.7152*lin[...,1]+0.2126*lin[...,2]
    h,w=Y.shape; s=cv2.resize(np.log(Y+1e-4).astype(np.float32),(w//4,h//4),interpolation=cv2.INTER_AREA)
    for _ in range(2): s=cv2.bilateralFilter(s,15,0.6,12)
    base=np.exp(cv2.resize(s,(w,h),interpolation=cv2.INTER_CUBIC))
    gain=np.clip((target/np.maximum(base,1e-4))**alpha,1.0,gmax)
    # fade gain in bright areas to protect windows/lamps
    gain=1+(gain-1)*np.clip((0.55-base)/0.35,0,1)
    return lin*gain[...,None]
def finish(img8, clahe_amt, vib, contrast):
    lab=cv2.cvtColor(img8,cv2.COLOR_BGR2LAB).astype(np.float32)
    L=lab[...,0]/255
    L=L+contrast*np.sin(2*np.pi*(L-0.5))*-0.5/np.pi*0+contrast*(L-0.5)*(1-np.abs(2*L-1))  # mild midtone contrast
    L8=np.clip(L*255,0,255).astype(np.uint8)
    cl=cv2.createCLAHE(clipLimit=1.4,tileGridSize=(8,8)).apply(L8).astype(np.float32)
    lab[...,0]=L8*(1-clahe_amt)+cl*clahe_amt
    a=lab[...,1]-128; b=lab[...,2]-128; c=np.sqrt(a*a+b*b)
    f=1+vib*(1-np.clip(c/40,0,1)); lab[...,1]=128+a*f; lab[...,2]=128+b*f
    return cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
P={ # upright, wb str, warm, exposure, local target(lin), alpha, gmax, clahe, vib, contrast
 1:(0.35,0.9,1.05,1.12,0.30,0.55,1.9,0.25,0.10,0.18),
 2:(0,  0.85,1.04,1.00,0.28,0.50,1.7,0.20,0.06,0.15),
 3:(0,  0.85,1.04,1.00,0.28,0.45,1.6,0.20,0.00,0.15),
 4:(0,  0.85,1.04,1.00,0.28,0.45,1.6,0.20,0.00,0.15),
 5:(0,  0.9,1.05,1.05,0.30,0.55,1.9,0.25,0.10,0.18),
}
def run(i,out_w=4000):
    fr,ws,warm,ex,tg,al,gm,cla,vib,con=P[i]
    im=cv2.imread(D+f'{i}.jpg')
    if fr: im,_,_=upright(im,fr)
    pass
    lin=srgb2lin(im.astype(np.float32)/255)
    lin,g=wb(lin,ws,warm); lin=lin*ex
    lin=local_tone(lin,tg,al,gm)
    # soft shoulder
    Y=0.0722*lin[...,0]+0.7152*lin[...,1]+0.2126*lin[...,2]; k=0.85
    Yc=np.where(Y>k,k+(1-k)*np.tanh((Y-k)/(1-k)),Y); lin=lin*(Yc/(Y+1e-6))[...,None]
    im=(lin2srgb(lin)*255+0.5).astype(np.uint8)
    im=finish(im,cla,vib,con)
    h,w=im.shape[:2]; im=cv2.resize(im,(out_w,int(out_w*h/w)),interpolation=cv2.INTER_LANCZOS4)
    return sharpen(im,0.35,1.5)
if __name__=='__main__':
    import sys
    for i in map(int,sys.argv[1:]):
        o=run(i); cv2.imwrite(f'out/Hotel_Calacoto_{i:02d}.jpg',o,[cv2.IMWRITE_JPEG_QUALITY,93])
        src=cv2.imread(D+f'{i}.jpg')
        cv2.imwrite(f'cmp{i}.jpg',np.hstack([cv2.resize(src,(1000,750)),cv2.resize(o,(1000,750),interpolation=cv2.INTER_AREA)]))
