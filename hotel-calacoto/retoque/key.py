import cv2, numpy as np
D='/tmp/claude-0/-home-user-cocoonnect/0ce6823a-e6f0-58ed-97c0-73d51fb19074/images/'
def segs(im):
    g=cv2.cvtColor(im,cv2.COLOR_BGR2GRAY); e=cv2.Canny(cv2.GaussianBlur(g,(5,5),0),50,150)
    L=cv2.HoughLinesP(e,1,np.pi/720,80,minLineLength=120,maxLineGap=8).reshape(-1,4)
    out=[]
    for x1,y1,x2,y2 in L:
        a=np.degrees(np.arctan2(x2-x1,y2-y1)); a=(a+90)%180-90
        if abs(a)<30: out.append((x1,y1,x2,y2,np.hypot(x2-x1,y2-y1),a))
    return np.array(out)
def vp_of(s):
    A=[];B=[];W=[]
    for x1,y1,x2,y2,l,a in s:
        n=np.array([y2-y1,-(x2-x1)],float); n/=np.linalg.norm(n); A.append(n);B.append(n@[x1,y1]);W.append(np.sqrt(l))
    A=np.array(A);B=np.array(B);W=np.array(W)
    return np.linalg.lstsq(A*W[:,None],B*W,rcond=None)[0]
def keystone(im,vp,frac):
    h,w=im.shape[:2]; c=np.array([w/2,h/2]); v=np.array(vp)-c
    T=np.array([[1,0,-c[0]],[0,1,-c[1]],[0,0,1.]]); Ti=np.linalg.inv(T)
    h3=-v/(v@v)*frac
    P=np.array([[1,0,0],[0,1,0],[h3[0],h3[1],1.]])
    # residual direction: map vp partially; shear to make center vertical straight
    H=Ti@P@T
    # add rotation so line through center & vp becomes vertical (roll fix)
    vpp=H@np.array([vp[0],vp[1],1.]); 
    d=(vpp[:2]/vpp[2]-c) if abs(vpp[2])>1e-9 else vpp[:2]
    if d[1]<0: d=-d
    th=np.arctan2(d[0],d[1])
    R=np.array([[np.cos(th),-np.sin(th),0],[np.sin(th),np.cos(th),0],[0,0,1]])
    H=Ti@R@T@H
    # normalize so corners fit: scale about center to keep bottom edge width
    out=cv2.warpPerspective(im,H,(w,h),flags=cv2.INTER_LANCZOS4)
    m=cv2.warpPerspective(np.full((h,w),255,np.uint8),H,(w,h),flags=cv2.INTER_NEAREST)
    return H,out,m
def bestcrop(m,w,h):
    for s in np.linspace(1,0.5,201):
        cw,ch=int(w*s),int(h*s); x0=(w-cw)//2
        for y0 in range(h-ch,-1,-4):  # prefer lower crops (keep bed/floor)
            if m[y0:y0+ch,x0:x0+cw].min()==255: return s,(x0,y0,cw,ch)
if __name__=='__main__':
    import sys
    for i in (1,4,5):
        im=cv2.imread(D+f'{i}.jpg'); s=segs(im); vp=vp_of(s)
        for frac in (0.4,0.6,0.8):
            H,o,m=keystone(im,vp,frac)
            # fit: scale image so content covers frame -> just measure crop
            sc,box=bestcrop(m,2000,1500); x,y,cw,ch=box
            r=segs(o[y:y+ch,x:x+cw]); print(i,frac,'crop',round(sc,3),'resid med',np.median(r[:,5]).round(2),'mean|a|',np.mean(np.abs(r[:,5])).round(2))
