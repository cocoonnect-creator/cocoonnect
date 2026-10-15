# Receta de retoque Hotel Calacoto (hotel-arenal/retoque/edit2.py) + correccion de perspectiva en TODAS las fotos:
# verticales corregidas (keystone hacia el punto de fuga vertical), recorte que conserva el 16:9 original
# y reduce el estiramiento de gran angular en los bordes. Salida 4K 3840x2160.
import cv2, numpy as np, sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'hotel-arenal', 'retoque'))
from key import segs, vp_of, keystone
from edit import srgb2lin, lin2srgb, wb, sharpen
from edit2 import local_tone, finish

def upright(im, frac, crop_left=0.0, zoom=1.0, anchor=0.5, ar=0.75):
    """Endereza verticales y recorta el mayor rectangulo valido de aspecto ar (alto/ancho)."""
    h,w=im.shape[:2]
    sg=segs(im); vp=vp_of(sg[np.abs(sg[:,5])<12])  # solo lineas casi verticales
    H,_,_=keystone(im,vp,frac)
    cn=np.array([[0,0],[w,0],[w,h],[0,h]],float).reshape(-1,1,2)
    q=cv2.perspectiveTransform(cn,H).reshape(-1,2); mn=q.min(0); mx=q.max(0)
    S=min(2*w/(mx-mn)[0],2*h/(mx-mn)[1])
    H2=np.array([[S,0,-mn[0]*S],[0,S,-mn[1]*S],[0,0,1]])@H
    W,Hh=int((mx-mn)[0]*S)+1,int((mx-mn)[1]*S)+1
    m=cv2.warpPerspective(np.full((h,w),255,np.uint8),H2,(W,Hh),flags=cv2.INTER_NEAREST)
    m=cv2.erode(m,np.ones((5,5),np.uint8))
    if crop_left:  # excluir contenido del borde izquierdo (p.ej. reflejo del fotografo)
        ml=cv2.warpPerspective(np.full((h,w),255,np.uint8)*0,H2,(W,Hh))
        x0=cv2.perspectiveTransform(np.array([[[w*crop_left,0]],[[w*crop_left,h]]],float),H2).reshape(-1,2)[:,0].max()
        m[:,:int(x0)]=0
    ii=cv2.integral((m==255).astype(np.uint8))
    full=lambda x,y,cw,ch: ii[y+ch,x+cw]-ii[y,x+cw]-ii[y+ch,x]+ii[y,x]==cw*ch
    lo,hi=0.2,1.0; sol=None
    for _ in range(14):
        md=(lo+hi)/2; cw,ch=int(W*md),int(W*md*ar)
        if ch>=Hh: hi=md; continue
        cand=[(x,y) for y in range(0,Hh-ch,6) for x in range(0,W-cw,6) if full(x,y,cw,ch)]
        if cand: lo=md; sol=(cand,cw,ch)
        else: hi=md
    cand,cw,ch=sol
    # anchor: 0 = recorte lo mas arriba posible (ventana, cabecero), 1 = lo mas abajo
    ys=sorted({c[1] for c in cand}); cy=ys[int(round(anchor*(len(ys)-1)))]
    xs=[c[0] for c in cand if c[1]==cy]; cx=np.mean(xs)
    # zoom>1 recorta mas apretado alrededor del centro valido (menos estiramiento en los bordes)
    x=cx+cw*(1-1/zoom)/2; y=cy+ch*(1-1/zoom)/2; cw/=zoom; ch/=zoom
    h=int(w*ar); Hf=np.diag([w/cw,h/ch,1])@np.array([[1,0,-x],[0,1,-y],[0,0,1.]])@H2
    print('  recorte conserva',round(cw*zoom/W*(mx-mn)[0]/w*S/S,3),'del ancho corregido')
    return cv2.warpPerspective(im,Hf,(w,h),flags=cv2.INTER_LANCZOS4,borderMode=cv2.BORDER_REFLECT), vp

P={ # frac verticales, crop izq, zoom, anchor, wb str, warm, exposicion, local target, alpha, gmax, clahe, vib, contraste
 1:(0.70,0.10,1.00,0.0, 0.85,1.04,1.00,0.28,0.45,1.6,0.20,0.04,0.15),  # bano: sacar reflejo del fotografo
 2:(0.75,0.0, 1.00,0.5, 0.85,1.04,1.00,0.28,0.45,1.6,0.20,0.04,0.15),
 3:(0.60,0.0, 1.00,0.0, 0.85,1.04,1.00,0.28,0.45,1.6,0.20,0.04,0.15),
 4:(0.60,0.0, 1.00,0.0, 0.85,1.04,1.00,0.28,0.45,1.6,0.20,0.04,0.15),
 5:(0.60,0.0, 1.00,0.0, 0.85,1.04,1.00,0.28,0.45,1.6,0.20,0.04,0.15),
}
def run(D,i,out=(4000,3000)):
    fr,cl,zm,an,ws,warm,ex,tg,al,gm,cla,vib,con=P[i]
    im=cv2.imread(D+f'{i}.jpg')
    im,vp=upright(im,fr,cl,zm,an)
    lin=srgb2lin(im.astype(np.float32)/255)
    lin,g=wb(lin,ws,warm); lin=local_tone(lin*ex,tg,al,gm)
    Y=0.0722*lin[...,0]+0.7152*lin[...,1]+0.2126*lin[...,2]; k=0.85
    lin=lin*(np.where(Y>k,k+(1-k)*np.tanh((Y-k)/(1-k)),Y)/(Y+1e-6))[...,None]
    im=finish((lin2srgb(lin)*255+0.5).astype(np.uint8),cla,vib,con)
    im=cv2.resize(im,out,interpolation=cv2.INTER_LANCZOS4)
    print(i,'vp',np.round(vp),'gains',g.round(3))
    return sharpen(im,0.30,1.5)
if __name__=='__main__':
    D,name,odir=sys.argv[1],sys.argv[2],sys.argv[3]
    for i in map(int,sys.argv[4:]):
        o=run(D,i); cv2.imwrite(f'{odir}/{name}_{i:02d}.jpg',o,[cv2.IMWRITE_JPEG_QUALITY,93])
        src=cv2.imread(D+f'{i}.jpg')
        cv2.imwrite(f'{odir}/../cmp{i}.jpg',np.hstack([cv2.resize(src,(1067,600)),cv2.resize(o,(800,600),interpolation=cv2.INTER_AREA)]))
