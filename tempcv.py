
from PIL import Image
from pylab import *
import sys
from PCV.localdescriptors import sift


if len(sys.argv) >= 3:
  im1f, im2f = sys.argv[1], sys.argv[2]
else:
  im1f = r'H:\UESTC\HomeWork\高级计算机视觉\homework\1.jpg'
  im2f = r'H:\UESTC\HomeWork\高级计算机视觉\homework\2.jpg'
im1 = array(Image.open(im1f))
im2 = array(Image.open(im2f))

sift.process_image(im1f, 'out_sift_1.txt')
l1, d1 = sift.read_features_from_file('out_sift_1.txt')
figure()
gray()
subplot(121)
sift.plot_features(im1, l1, circle=False)

sift.process_image(im2f, 'out_sift_2.txt')
l2, d2 = sift.read_features_from_file('out_sift_2.txt')
subplot(122)
sift.plot_features(im2, l2, circle=False)

#matches = sift.match(d1, d2)
matches = sift.match_twosided(d1, d2)
print( '{} matches'.format(len(matches.nonzero()[0])))

figure()
gray()
sift.plot_matches(im1, im2, l1, l2, matches, show_below=True)
show()





# # -*- coding: utf-8 -*-
# from PIL import Image
# from pylab import *
# from PCV.localdescriptors import sift
# from PCV.localdescriptors import harris
#
# # 添加中文字体支持
# from matplotlib.font_manager import FontProperties
# font = FontProperties(fname=r"c:\windows\fonts\SimSun.ttc", size=14)
#
# imname = r'H:\UESTC\HomeWork\高级计算机视觉\homework\1.jpg'
# im = array(Image.open(imname).convert('L'))
# sift.process_image(imname, 'empire.sift')
# l1, d1 = sift.read_features_from_file('empire.sift')
#
# figure()
# gray()
# subplot(131)
# sift.plot_features(im, l1, circle=False)
# title(u'SIFT特征',fontproperties=font)
# subplot(132)
# sift.plot_features(im, l1, circle=True)
# title(u'用圆圈表示SIFT特征尺度',fontproperties=font)
#
# # 检测harris角点
# harrisim = harris.compute_harris_response(im)
#
# subplot(133)
# filtered_coords = harris.get_harris_points(harrisim, 6, 0.1)
# imshow(im)
# plot([p[1] for p in filtered_coords], [p[0] for p in filtered_coords], '*')
# axis('off')
# title(u'Harris角点',fontproperties=font)
#
# show()
