import loader
import pathfinder

subject = 'Dog_1'
test_clip2label, test_clip2time, test_clip2usage = loader.load_test_labels(pathfinder.LABELS_PATH)
test_preictal_groups, test_interictal_groups = loader.group_labels_by_hour(test_clip2label, test_clip2time, subject)

for g in test_preictal_groups:
    usages = [test_clip2usage[c] for c in g]
    print usages

print 'Interictal'
for g in test_interictal_groups:
    usages = [test_clip2usage[c] for c in g]
    print usages