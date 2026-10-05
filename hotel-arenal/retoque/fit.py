from key import *
def upright(im,frac,vp=None):
    h,w=im.shape[:2]
    if vp is None: vp=vp_of(segs(im))
    H,_,_=keystone(im,vp,frac)
    cn=np.array([[0,0],[w,0],[w,h],[0,h]],float).reshape(-1,1,2)
    q=cv2.perspectiveTransform(cn,H).reshape(-1,2)
    mn=q.min(0); mx=q.max(0); S=min(2*w/(mx-mn)[0],2*h/(mx-mn)[1])
    A=np.array([[S,0,-mn[0]*S],[0,S,-mn[1]*S],[0,0,1]]); H2=A@H
    W,Hh=int((mx-mn)[0]*S)+1,int((mx-mn)[1]*S)+1
    m=cv2.warpPerspective(np.full((h,w),255,np.uint8),H2,(W,Hh),flags=cv2.INTER_NEAREST)
    m=cv2.erode(m,np.ones((5,5),np.uint8))
    # largest 4:3 rect: binary search scale, grid search position
    ii=cv2.integral((m==255).astype(np.uint8))
    def ok(cw,ch):
        best=None
        for y in range(0,Hh-ch,6):
            for x in range(0,W-cw,6):
                if ii[y+ch,x+cw]-ii[y,x+cw]-ii[y+ch,x]+ii[y,x]==cw*ch:
                    if best is None or y>best[1] or (y==best[1]): best=(x,y)
            # keep scanning for lower positions
        return best
    lo,hi=0.2,1.0; sol=None
    for _ in range(14):
        md=(lo+hi)/2; cw,ch=int(W*md),int(W*md*0.75)
        if ch>=Hh: hi=md;continue
        r=ok(cw,ch)
        if r: lo=md; sol=(r,cw,ch)
        else: hi=md
    (x,y),cw,ch=sol
    # center horizontally among valid x at chosen y
    xs=[xx for xx in range(0,W-cw,2) if ii[y+ch,xx+cw]-ii[y,xx+cw]-ii[y+ch,xx]+ii[y,xx]==cw*ch]
    x=xs[len(xs)//2]
    T=np.array([[1,0,-x],[0,1,-y],[0,0,1.]]); Sc=np.diag([w/cw,h/ch,1])
    Hf=Sc@T@H2
    out=cv2.warpPerspective(im,Hf,(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT)
    return out,Hf,w/cw*S
if __name__=='__main__':
    for i,fr in ((1,0.35),(5,0.35),(4,0.3)):
        im=cv2.imread(D+f'{i}.jpg'); o,Hf,z=upright(im,fr); print(i,'zoom',round(z,3))
        cv2.imwrite(f'k{i}.jpg',np.hstack([cv2.resize(im,(1000,750)),cv2.resize(o,(1000,750))]))
