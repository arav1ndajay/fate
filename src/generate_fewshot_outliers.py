import random

class GenerateFewShotOutliers():

  def __init__(self, dataset: str, subset: str) -> None:
    self.subsets= {
      "ag": ["business", "sci", "sports", "world"],
      "20ng": ["comp", "rec", "sci", "misc", "pol", "rel"],
      "reuters": ["earn", "acq", "crude", "trade", "money-fx", "interest", "ship"]
    }

    # choose inlier dataset and subset
    self.dataset = dataset
    self.subset = subset

    # k in k-shot
    self.num_outliers = 10
    print(f"Inlier dataset: {self.dataset}/{self.subset}")

    base_file_path = f"../datasets/{dataset}/train/"
    outlier_sentences = []
    all_possible_outlier_sentences = []

    for sset in self.subsets[dataset]:
        
        # special case for CLINIC150 as it does not have subsets
        if self.dataset == "clinic150_od":
          sentences = self.get_sentences(base_file_path + sset + ".txt")
          outlier_sentences.extend(random.sample(sentences, self.num_outliers))
          break
    
        if len(outlier_sentences) >= self.num_outliers:
          break

        if sset != self.subset:
            print("Getting anomalies from : ", base_file_path + sset)
            sentences = self.get_sentences(base_file_path + sset + ".txt")

            outlier_sentences.extend(random.sample(sentences, int(self.num_outliers / (len(self.subsets[dataset]) - 1))))
            
            # ensure no overlap between outliers and all sentences
            filtered_sentences = [i for i in sentences if i not in outlier_sentences]
            all_possible_outlier_sentences.extend(filtered_sentences)

    if len(outlier_sentences) < self.num_outliers:
      outlier_sentences.extend(random.sample(all_possible_outlier_sentences, self.num_outliers - len(outlier_sentences)))

    # writing outliers to file
    file = open(base_file_path + subset + "-outliers.txt",'w')
    for sent in outlier_sentences:
      file.write(sent+"\n\n")
    file.close()

  def get_sentences(self, file_path):
      with open(file_path, encoding='utf-8') as f:
          sentences = [
          line 
          for line in f.read().splitlines()
          if (len(line) > 0 and not line.isspace())
          ]
      
      return sentences