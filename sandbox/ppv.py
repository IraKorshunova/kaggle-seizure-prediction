# sn = 0.74
# fpr = 0.06
# sp = 1. - fpr
#
# npre = 3.125
# nint = 54.6 - 3.125
#
# tp = sn * npre
# tn = sp * nint
# fp = nint - tn
# ppv = tp / (tp + fp)
# print ppv

sn = 0.46
fpd = 0.64
nh = 213 * 24
npre = 18


fpr = 0.9 / 24
sp = 1. - fpr


nint = nh - npre

tp = sn * npre
tn = sp * nint
fp = nint - tn
ppv = tp / (tp + fp)
print ppv
