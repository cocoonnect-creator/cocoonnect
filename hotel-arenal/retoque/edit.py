import cv2, numpy as np
from fit import upright, D
def srgb2lin(x): return np.where(x<=0.04045,x/12.92,((x+0.055)/1.055)**2.4)
def lin2srgb(x): x=np.clip(x,0,1); return np.where(x<=0.0031308,x*12.92,1.055*x**(1/2.4)-0.055)
def wb(lin,strength,warm):
    s=lin2srgb(lin); mx=s.max(2); mn=s.min(2); sat=(mx-mn)/(mx+1e-6); l=s.mean(2)
    m=(sat<0.25)&(l>0.35)&(mx<0.97)
    mean=lin[m].mean(0)  # BGR
    g=mean[1]/mean; tgt=np.array([1/warm,1.0,warm])  # BGR target ratio
    gains=g*tgt; gains/=gains[1]
    gains=1+(gains-1)*strength
    return lin*gains, gains
def tone(lin,target_mid,hi_knee=0.80):
    # exposure to target median luminance (in sRGB space)
    Y=(0.0722*lin[...,0]+0.7152*lin[...,1]+0.2126*lin[...,2])
    med=np.median(lin2srgb(Y)); ev=srgb2lin(np.array(target_mid))/srgb2lin(np.array(med))
    ev=float(np.clip(ev,1,2.2)); lin=lin*ev
    # highlight shoulder (protect lamps/windows) on luminance
    Y=(0.0722*lin[...,0]+0.7152*lin[...,1]+0.2126*lin[...,2])
    k=hi_knee; Yc=np.where(Y>k, k+(1-k)*(1-np.exp(-(Y-k)/(1-k))), Y)
    lin=lin*(Yc/(Y+1e-6))[...,None]
    return lin, ev
def finish(img8, clahe_amt, vib, shadow):
    lab=cv2.cvtColor(img8,cv2.COLOR_BGR2LAB).astype(np.float32)
    L=lab[...,0]/255
    # shadow lift
    L2=L+shadow*(1-L)*np.exp(-((L)/0.30)**2)*L*2
    # gentle S contrast
    L2=L2+0.06*np.sin(np.pi*(L2-0.5))*0.5*(L2>0)
    L8=np.clip(L2*255,0,255).astype(np.uint8)
    cl=cv2.createCLAHE(clipLimit=1.6,tileGridSize=(8,8)).apply(L8).astype(np.float32)
    lab[...,0]=np.clip(L8*(1-clahe_amt)+cl*clahe_amt,0,255)
    a=lab[...,1]-128; b=lab[...,2]-128; c=np.sqrt(a*a+b*b)
    f=1+vib*(1-np.clip(c/40,0,1))
    lab[...,1]=128+a*f; lab[...,2]=128+b*f
    return cv2.cvtColor(np.clip(lab,0,255).astype(np.uint8),cv2.COLOR_LAB2BGR)
def sharpen(img,amt,r):
    bl=cv2.GaussianBlur(img,(0,0),r)
    d=img.astype(np.float32)-bl
    d=np.where(np.abs(d)<2,0,d)  # threshold to avoid noise
    return np.clip(img+amt*d,0,255).astype(np.uint8)
P={ # id: (upright frac, wb strength, warm, target mid, clahe, vibrance, shadow, knee)
 1:(0.35,0.85,1.035,0.66,0.30,0.10,0.10,0.80),
 2:(0,  0.80,1.03, 0.64,0.30,0.08,0.10,0.75),
 3:(0,  0.60,1.03, 0.62,0.25,0.00,0.08,0.75),
 4:(0,  0.60,1.03, 0.62,0.25,0.00,0.08,0.75),
 5:(0,  0.85,1.035,0.66,0.30,0.10,0.10,0.80),
}
def run(i, out_w=4000):
    fr,ws,warm,tm,cla,vib,sh,knee=P[i]
    im=cv2.imread(D+f'{i}.jpg')
    if fr: im,_,_=upright(im,fr)
    im=cv2.fastNlMeansDenoisingColored(im,None,3,3,7,21)
    lin=srgb2lin(im.astype(np.float32)/255)
    lin,g=wb(lin,ws,warm); lin,ev=tone(lin,tm,knee)
    im=(lin2srgb(lin)*255+0.5).astype(np.uint8)
    im=finish(im,cla,vib,sh)
    h,w=im.shape[:2]; im=cv2.resize(im,(out_w,int(out_w*h/w)),interpolation=cv2.INTER_LANCZOS4)
    im=sharpen(im,0.45,1.6)
    print(i,'gains BGR',g.round(3),'ev',round(ev,2))
    return im
if __name__=='__main__':
    import sys
    for i in map(int,sys.argv[1:]):
        o=run(i); cv2.imwrite(f'out/Hotel_Calacoto_{i:02d}.jpg',o,[cv2.IMWRITE_JPEG_QUALITY,93])
        src=cv2.imread(D+f'{i}.jpg')
        cv2.imwrite(f'cmp{i}.jpg',np.hstack([cv2.resize(src,(1000,750)),cv2.resize(o,(1000,750),interpolation=cv2.INTER_AREA)]))
