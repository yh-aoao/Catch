from pathlib import Path,PurePosixPath
from zipfile import ZipFile,ZIP_DEFLATED
from PIL import Image
import io,math,posixpath,json
import xml.etree.ElementTree as X
src=Path('/home/isee/Catch/analysis_artifacts/two_stage_module_20261126/output/论文模型图.pptx')
dst=Path('/home/isee/Catch/analysis_artifacts/ppt_script_20261128/output/论文模型图_统一模块底色框.pptx')
A='http://schemas.openxmlformats.org/drawingml/2006/main';P='http://schemas.openxmlformats.org/presentationml/2006/main';R='http://schemas.openxmlformats.org/officeDocument/2006/relationships';PKG='http://schemas.openxmlformats.org/package/2006/relationships';ns={'a':A,'p':P}
for k,v in [('a',A),('p',P),('r',R)]:X.register_namespace(k,v)
with ZipFile(src) as z:files={i.filename:z.read(i.filename) for i in z.infolist()}
report={'cropped':[],'removed_duplicates':[],'group_boxes':[],'unfilled_groups_skipped':[3,5,6]}
roots={}
for si in [0,3,4,5]:
 name=f'ppt/slides/slide{si+1}.xml'
 if name not in files:continue
 root=X.fromstring(files[name]);roots[name]=root;tree=root.find('p:cSld/p:spTree',ns)
 relroot=X.fromstring(files[f'ppt/slides/_rels/slide{si+1}.xml.rels'])
 rels={e.get('Id'):posixpath.normpath(posixpath.join('ppt/slides',e.get('Target'))) for e in relroot}
 # Remove the requested duplicate layers, plus matching image/crop/frame duplicates
 # when PowerPoint has reassigned the shape IDs.
 if si==0:
  seen=[]
  for pic in list(tree.findall('p:pic',ns)):
   ident=pic.find('p:nvPicPr/p:cNvPr',ns);idx=int(ident.get('id'))
   blip=pic.find('p:blipFill/a:blip',ns);crop=pic.find('p:blipFill/a:srcRect',ns);xf=pic.find('p:spPr/a:xfrm',ns)
   sig=(rels[blip.get('{'+R+'}embed')],tuple(int(crop.get(k,'0')) if crop is not None else 0 for k in ['l','t','r','b']))
   geom=tuple(int(v) for tag in ['off','ext'] for v in xf.find('a:'+tag,ns).attrib.values())
   dup=any(sig==s and all(abs(a-b)<=9525 for a,b in zip(geom,g)) for s,g in seen)
   if idx in (85,86) or dup:tree.remove(pic);report['removed_duplicates'].append({'slide':1,'shape_id':idx})
   else:seen.append((sig,geom))
 for pic in tree.findall('p:pic',ns):
  blip=pic.find('p:blipFill/a:blip',ns);path=rels[blip.get('{'+R+'}embed')]
  img=Image.open(io.BytesIO(files[path])).convert('RGB');W,H=img.size
  fill=pic.find('p:blipFill',ns);crop=fill.find('a:srcRect',ns)
  if crop is None:crop=X.Element('{'+A+'}srcRect');fill.insert(1,crop)
  l,t,r,b=[int(crop.get(k,'0'))/100000 for k in ['l','t','r','b']]
  left,top,right,bottom=round(l*W),round(t*H),round((1-r)*W),round((1-b)*H)
  if left<0 or top<0 or right>W or bottom>H or right<=left or bottom<=top:continue
  visible=img.crop((left,top,right,bottom));mask=visible.convert('L').point(lambda v:255 if v<245 else 0);bbox=mask.getbbox()
  if bbox is None:continue
  x0,y0,x1,y1=bbox;pad=round(min(visible.size)*.01)
  x0=max(0,x0-pad);x1=min(visible.width,x1+pad);y0=max(0,y0-pad);y1=min(visible.height,y1+pad)
  # Preserve the existing visible aspect ratio, avoiding stretched icons.
  ratio=visible.width/visible.height;cw=x1-x0;ch=y1-y0
  if cw/ch>ratio:ch=cw/ratio
  else:cw=ch*ratio
  cx=(x0+x1)/2;cy=(y0+y1)/2
  x0=max(0,min(visible.width-cw,cx-cw/2));y0=max(0,min(visible.height-ch,cy-ch/2));x1=x0+cw;y1=y0+ch
  vals={'l':round((left+x0)/W*100000),'t':round((top+y0)/H*100000),'r':round((1-(left+x1)/W)*100000),'b':round((1-(top+y1)/H)*100000)}
  old=dict(crop.attrib)
  for k,v in vals.items():crop.set(k,str(max(0,min(100000,v))))
  report['cropped'].append({'slide':si+1,'shape_id':pic.find('p:nvPicPr/p:cNvPr',ns).get('id'),'old':old,'new':vals})
# Only the two supplied concrete ID lists are actionable. Ellipsis placeholders stay skipped.
for si,ids in {0:[47,48,59,63],1:list(range(2,13))}.items():
 name=f'ppt/slides/slide{si+1}.xml';root=roots.get(name)
 if root is None:root=X.fromstring(files[name]);roots[name]=root
 tree=root.find('p:cSld/p:spTree',ns);rects=[]
 for el in list(tree)[2:]:
  ident=el.find('.//p:cNvPr',ns)
  if ident is None or int(ident.get('id')) not in ids:continue
  xf=el.find('p:spPr/a:xfrm',ns)
  if xf is None:xf=el.find('p:grpSpPr/a:xfrm',ns)
  if xf is None:continue
  off=xf.find('a:off',ns);ext=xf.find('a:ext',ns);rects.append(tuple(int(v) for v in [off.get('x'),off.get('y'),ext.get('cx'),ext.get('cy')]))
 if len(rects)<2:continue
 pad=round(914400*.12);x=min(v[0] for v in rects)-pad;y=min(v[1] for v in rects)-pad;w=max(v[0]+v[2] for v in rects)-x+pad;h=max(v[1]+v[3] for v in rects)-y+pad
 idx=max(int(e.get('id')) for e in root.findall('.//p:cNvPr',ns))+1
 sp=X.Element('{'+P+'}sp');nv=X.SubElement(sp,'{'+P+'}nvSpPr');X.SubElement(nv,'{'+P+'}cNvPr',{'id':str(idx),'name':'统一模块底色框'});X.SubElement(nv,'{'+P+'}cNvSpPr');X.SubElement(nv,'{'+P+'}nvPr');pr=X.SubElement(sp,'{'+P+'}spPr');xf=X.SubElement(pr,'{'+A+'}xfrm');X.SubElement(xf,'{'+A+'}off',{'x':str(x),'y':str(y)});X.SubElement(xf,'{'+A+'}ext',{'cx':str(w),'cy':str(h)});gg=X.SubElement(pr,'{'+A+'}prstGeom',{'prst':'roundRect'});X.SubElement(gg,'{'+A+'}avLst');f=X.SubElement(pr,'{'+A+'}solidFill');X.SubElement(f,'{'+A+'}srgbClr',{'val':'E8EFF7'});ln=X.SubElement(pr,'{'+A+'}ln',{'w':'12700'});f=X.SubElement(ln,'{'+A+'}solidFill');X.SubElement(f,'{'+A+'}srgbClr',{'val':'9AB3CC'});X.SubElement(pr,'{'+A+'}effectLst');tree.insert(2,sp)
 report['group_boxes'].append({'slide':si+1,'ids':ids})
for name,root in roots.items():files[name]=X.tostring(root,encoding='utf-8',xml_declaration=True)
with ZipFile(dst,'w',ZIP_DEFLATED) as z:
 for name,data in files.items():z.writestr(name,data)
Path('/home/isee/Catch/analysis_artifacts/ppt_script_20261128/build/report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(dst);print('Cropped:',len(report['cropped']),'Duplicates removed:',report['removed_duplicates'],'Group boxes:',len(report['group_boxes']))
