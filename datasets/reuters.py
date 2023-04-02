import nltk
from nltk import word_tokenize
from nltk.corpus import reuters
import os
from clean_text import clean_text

directory = "./reuters"
nltk.download('reuters', download_dir=directory)
if directory not in nltk.data.path:
  nltk.data.path.append(directory)

def write_to_file(sentences, file_path):
    print(f"Writing sentences to {file_path}...")
    file = open(file_path, 'w')
    for sent in sentences:
      file.write(sent+"\n\n")
    file.close()

doc_ids = reuters.fileids()

ret = []

train = True
test = True
splits = [split_set for (requested, split_set) in [(train, 'train'), (test, 'test')] if requested]

for split_set in splits:
  split_set_doc_ids = list(filter(lambda doc: doc.startswith(split_set), doc_ids))
  examples = []

  clean_txt = True

  for id in split_set_doc_ids:
      if clean_txt:
          text = clean_text(reuters.raw(id))
      else:
          text = ' '.join(word_tokenize(reuters.raw(id)))
      labels = reuters.categories(id)

      examples.append({
          'text': text,
          'label': labels,
      })
  
  ret.append(examples)

train_set, test_set = ret
print(len(train_set), len(test_set))

# all classes
classes = ['earn', 'acq', 'crude', 'trade', 'money-fx', 'interest', 'ship']

classwise_train = {}
classwise_test = {}

for cls in classes:
  classwise_train[cls] = []
  classwise_test[cls] = []

# segregate training instances class-wise
for example in train_set:
    cls, text = example['label'], example['text']
    if len(cls) == 1 and cls[0] in classes:
        classwise_train[cls[0]].append(text.lower())

# segregate testing instances class-wise   
for example in test_set:
    cls, text = example['label'], example['text']
    if len(cls) == 1 and cls[0] in classes:
        classwise_test[cls[0]].append(text.lower())

print("Training: ")
for key, value in classwise_train.items():
    print(key, ": ", len(value))

print("Testing: ")
for key, value in classwise_test.items():
    print(key, ": ", len(value))

# store in files
base_file_path = directory

if not os.path.exists(base_file_path + "/train"):
    os.mkdir(base_file_path + "/train")

if not os.path.exists(base_file_path + "/test"):
    os.mkdir(base_file_path + "/test")

# keep one class normal and rest as anomalies
for normal_class in classes:
    outlier_classes = [cls for cls in classes if cls != normal_class]
    train_inliers = classwise_train[normal_class]
    test_inliers = classwise_test[normal_class]

    train_outliers = []
    test_outliers = []
    for cls in outlier_classes:
        train_outliers.extend(classwise_train[cls])
        test_outliers.extend(classwise_test[cls])

    # setting up paths
    train_inlier_path = base_file_path + f"/train/{normal_class}.txt"
    train_outlier_path = base_file_path + f"/train/{normal_class}-outliers.txt"
    test_inlier_path = base_file_path + f"/test/{normal_class}.txt"
    test_outlier_path = base_file_path + f"/test/{normal_class}-outliers.txt"

    # writing to paths
    write_to_file(train_inliers, train_inlier_path)
    write_to_file(train_outliers, train_outlier_path)
    write_to_file(test_inliers, test_inlier_path)
    write_to_file(test_outliers, test_outlier_path)

print("Reuters dataset prepared.")
    