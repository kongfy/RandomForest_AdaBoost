package auxiliary;

import java.util.*;

/**
 *
 * @author 孔繁宇 MF1333020
 */

//存储可放回抽取生成的新样本集
class RepickSamples
{
	double[][] features;
	double[] labels;
}

public class RandomForest extends Classifier {
	private static int classifier = 3;      //生成分类器的数量
	private RandomDecisionTree forest[];
	
    public RandomForest() {
    }

    @Override
    public void train(boolean[] isCategory, double[][] features, double[] labels) {
    	forest = new RandomDecisionTree[classifier];
    	for (int i = 0; i < classifier; ++i) {
    		RepickSamples samples = repickSamples(features, labels);
    		forest[i] = new RandomDecisionTree();
    		forest[i].train(isCategory, samples.features, samples.labels);
    	}
    }
    
    //可放回收取新样本集
    private RepickSamples repickSamples(double[][] features, double[] labels) {
    	RepickSamples samples = new RepickSamples();
    	int size = labels.length;
    	Random random = new Random();
    	
    	samples.features = new double[size][];
    	samples.labels = new double[size];
    	for (int i = 0; i < size; ++i) {
    		int index = random.nextInt(size);
    		samples.features[i] = features[index].clone();
    		samples.labels[i] = labels[index];
    	}
    	
    	return samples;
    }

    @Override
    public double predict(double[] features) {
    	HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
        for (int i = 0; i < forest.length; ++i) {
        	double label = forest[i].predict(features);
        	if (counter.get(label) == null) {
        		counter.put(label, 1);
        	} else {
        		int count = counter.get(label);
        		counter.put(label, count++);
        	}
        }
        
        int temp_max = 0;
        double label = 0;
        Iterator<Double> iterator = counter.keySet().iterator();
        while (iterator.hasNext()) {
            double key = iterator.next();
            int count = counter.get(key);
            if (count > temp_max) {
                temp_max = count;
                label = key;
            }
        }
        
        return label;
    }
}

// <<<<--------------------------华丽的分界线，下面是随机决策树的实现---------------------------->>>>

//决策树节点结构
class TreeNode {
  int[] set;                         //样本下标集合
  int[] attr_index;                  //可用属性下标集合
  double label;                      //标签
  int split_attr;                    //该节点用于分割的属性下标
  double[] split_points;             //切割点 离散属性为多值，连续属性只有一个值
  TreeNode[] childrenNodes;          //子节点
}

//存储分割信息
class SplitData {
  int split_attr;
  double[] split_points;
  int[][] split_sets;                //分割后新的样本集合的数组
}

class BundleData {
  double floatValue;                 //存储增益率或MSE
  SplitData split_info;
}

//当分割出现错误时抛出此异常
class SplitException extends Exception {
}

class RandomDecisionTree extends Classifier {

  private boolean _isClassification;
  private double[][] _features;
  private boolean[] _isCategory;
  private double[] _labels;
  private double[] _defaults;
  
  private TreeNode root;

  public RandomDecisionTree() {
  }
  
  @Override
  public void train(boolean[] isCategory, double[][] features, double[] labels) {
      _isClassification = isCategory[isCategory.length - 1];
      _features = features;
      _isCategory = isCategory;
      _labels = labels;
      
      int set[] = new int[_features.length];
      for (int i = 0; i < set.length; ++i) {
          set[i] = i;
      }
      
      int attr_index[] = new int[_features[0].length];
      for (int i = 0; i < attr_index.length; ++i) {
          attr_index[i] = i;
      }
      
      //处理缺失属性
      _defaults = kill_missing_data();
      
      root = build_decision_tree(set, attr_index);
  }
  
  private double[] kill_missing_data() {
      int num = _isCategory.length - 1;
      double[] defaults = new double[num];
      
      for (int i = 0; i < defaults.length; ++i) {
          if (_isCategory[i]) {
              //离散属性取最多的值
              HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
              for (int j = 0; j < _features.length; ++j) {
                  double feature = _features[j][i];
                  if (!Double.isNaN(feature)) {
                      if (counter.get(feature) == null) {
                          counter.put(feature, 1);
                      } else {
                          int count = counter.get(feature) + 1;
                          counter.put(feature, count);
                      }
                  }
              }
              
              int max_time = 0;
              double value = 0;
              Iterator<Double> iterator = counter.keySet().iterator();
              while (iterator.hasNext()) {
                  double key = iterator.next();
                  int count = counter.get(key);
                  if (count > max_time) {
                      max_time = count;
                      value = key;
                  }
              }
              defaults[i] = value;
          } else {
              //连续属性取平均值
              int count = 0;
              double total = 0;
              for (int j = 0; j < _features.length; ++j) {
                  if (!Double.isNaN(_features[j][i])) {
                      count++;
                      total += _features[j][i];
                  }
              }
              defaults[i] = total / count;
          }
      }
      
      //代换
      for (int i = 0; i < _features.length; ++i) {
          for (int j = 0; j < defaults.length; ++j) {
              if (Double.isNaN(_features[i][j])) {
                  _features[i][j] = defaults[j];
              }
          }
      }
      return defaults;
  }

  @Override
  public double predict(double[] features) {
      //处理缺失属性
      for (int i = 0; i < features.length; ++i) {
          if (Double.isNaN(features[i])) {
              features[i] = _defaults[i];
          }
      }
      
      return predict_with_decision_tree(features, root);
  }
  
  private double predict_with_decision_tree(double[] features, TreeNode node) {
      if (node.childrenNodes == null) {
          return node.label;
      }
      
      double feature = features[node.split_attr];
      
      if (_isCategory[node.split_attr]) {
          //离散属性
          for (int i = 0; i < node.split_points.length; ++i) {
              if (node.split_points[i] == feature) {
                  return predict_with_decision_tree(features, node.childrenNodes[i]);
              }
          }
          
          return node.label; //不存在的属性取父节点样本的标签，减少叶子结点
      } else {
          //连续属性
          if (feature < node.split_points[0]) {
              return predict_with_decision_tree(features, node.childrenNodes[0]);
          } else {
              return predict_with_decision_tree(features, node.childrenNodes[1]);
          }
      }
      
  }
  
  private TreeNode build_decision_tree(int[] set, int[] attr_index) {
      TreeNode node = new TreeNode();
      node.set = set;
      node.attr_index = attr_index;
      node.label = 0;
      node.childrenNodes = null;
      
      //都为同类返回直接返回
      double label = _labels[node.set[0]];
      boolean flag = true;
      for (int i = 0; i < node.set.length; ++i) {
          if (_labels[node.set[i]] != label) {
              flag = false;
              break;
          }
      }
      if (flag) {
          node.label = label;
          return node;
      }
      
      //没有可用属性标记为大多数(离散)或平均值(连续)
      if (_isClassification) {
          node.label = most_label(set);
      } else {
          node.label = mean_value(set);
      }
      if (node.attr_index == null || node.attr_index.length == 0) {
          return node;
      }
      
      //寻找最优切割属性
      SplitData split_info = attribute_selection(node);
      node.split_attr = split_info.split_attr;
      //没有可以分割的属性
      if (node.split_attr < 0) {
          return node;
      }
      
      node.split_points = split_info.split_points;
      
      //去掉已使用的离散属性，连续属性不做删除
      int[] child_attr_index = null;
      if (_isCategory[node.split_attr]) {
          child_attr_index = new int[attr_index.length - 1];
          int t = 0;
          for (int index : attr_index) {
              if (index != node.split_attr) {
                  child_attr_index[t++] = index;
              }
          }
      } else {
          child_attr_index = node.attr_index.clone();
      }
      
      //递归建立子节点
      node.childrenNodes = new TreeNode[split_info.split_sets.length];
      for (int i = 0; i < split_info.split_sets.length; ++i) {
          node.childrenNodes[i] = build_decision_tree(split_info.split_sets[i], child_attr_index);
      }
      
      return node;
  }
  
  //给定样本中出现最多的标签
  private double most_label(int[] set) {
      HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
      for (int item : set) {
          double label = _labels[item];
          if (counter.get(label) == null) {
              counter.put(label, 1);
          } else {
              int count = counter.get(label) + 1;
              counter.put(label, count);
          }
      }
      
      int max_time = 0;
      double label = 0;
      Iterator<Double> iterator = counter.keySet().iterator();
      while (iterator.hasNext()) {
          double key = iterator.next();
          int count = counter.get(key);
          if (count > max_time) {
              max_time = count;
              label = key;
          }
      }
      return label;
  }
  
  //给定样本的标签平均值
  private double mean_value(int[] set) {
      double temp = 0;
      for (int index : set) {
          temp += _labels[index];
      }
      return temp / set.length;
  }
  
  private SplitData attribute_selection(TreeNode node) {
      SplitData result = new SplitData();
      result.split_attr = -1;
      
      //前剪枝
      double reference_value = _isClassification ? 0.05 : -1;
      if (node.set.length < 7) return result;
      
      //生成随机选取的属性
      int n = 3;
      int attrs[] = new int[n];
      Random random = new Random();
      for (int i = 0; i < n; ++i) {
    	  int index = random.nextInt(node.attr_index.length);
    	  attrs[i] = node.attr_index[index];
      }
      if (_isClassification) {
          for (int attribute : attrs) {
              try {
                  BundleData gain_ratio_info = gain_ratio_use_attribute(node.set, attribute); //分割错误会抛出分割异常
                  if (gain_ratio_info.floatValue > reference_value) {
                      reference_value = gain_ratio_info.floatValue;
                      result = gain_ratio_info.split_info;
                  }
              } catch (SplitException ex) { //捕获异常，直接丢弃
              }
          }
      } else {
          for (int attribute : attrs) {
              try {
                  BundleData mse_info = mse_use_attribute(node.set, attribute);
                  if (reference_value < 0 || mse_info.floatValue < reference_value) {
                      reference_value = mse_info.floatValue;
                      result = mse_info.split_info;
                  }
              } catch (SplitException ex) {
              }
          }
      }
      return result;
  }
  
  private SplitData split_with_attribute(int[] set, int attribute) throws SplitException {
      SplitData result = new SplitData();
      result.split_attr = attribute;
      
      if (_isCategory[attribute]) {
          //离散属性
          int amount_of_features = 0;
          HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
          HashMap<Double, Integer> index_recorder = new HashMap<Double, Integer>();
          for (int item : set) {
              double feature = _features[item][attribute];
              if (counter.get(feature) == null) {
                  counter.put(feature, 1);
                  index_recorder.put(feature, amount_of_features++);
              } else {
                  int count = counter.get(feature) + 1;
                  counter.put(feature, count);
              }
          }
          
          //记录切割点
          result.split_points = new double[amount_of_features];
          Iterator<Double> iterator = index_recorder.keySet().iterator();
          
          while (iterator.hasNext()) {
              double key = iterator.next();
              int value = index_recorder.get(key);
              result.split_points[value] = key;
          }
          
          result.split_sets = new int[amount_of_features][];
          int[] t_index = new int[amount_of_features];
          for (int i = 0; i < amount_of_features; ++i) t_index[i] = 0;
          
          for (int item : set) {
              int index = index_recorder.get(_features[item][attribute]);
              if (result.split_sets[index] == null) {
                  result.split_sets[index] = new int[counter.get(_features[item][attribute])];
              }
              result.split_sets[index][t_index[index]++] = item;
          }
      } else {
          //连续属性
          double[] features = new double[set.length];
          for (int i = 0; i < features.length; ++i) {
              features[i] = _features[set[i]][attribute];
          }
          Arrays.sort(features);
          
          double reference_value = _isClassification ? 0 : -1;
          double best_split_point = 0;
          result.split_sets = new int[2][];
          for (int i = 0; i < features.length - 1; ++i) {
              if (features[i] == features[i + 1]) continue;
              double split_point = (features[i] + features[i + 1]) / 2;
              int[] sub_set_a = new int[i + 1];
              int[] sub_set_b = new int[set.length - i - 1];
              
              int a_index = 0;
              int b_index = 0;
              for (int j = 0; j < set.length; ++j) {
                  if (_features[set[j]][attribute] < split_point) {
                      sub_set_a[a_index++] = set[j];
                  } else {
                      sub_set_b[b_index++] = set[j];
                  }
              }
              
              if (_isClassification) {
                  double temp = gain_ratio_use_numerical_attribute(set, attribute, sub_set_a, sub_set_b);
                  if (temp > reference_value) {
                      reference_value = temp;
                      best_split_point = split_point;
                      result.split_sets[0] = sub_set_a;
                      result.split_sets[1] = sub_set_b;
                  }
              } else {
                  double temp = (sub_set_a.length * mse(sub_set_a) + sub_set_b.length * mse(sub_set_b)) / set.length;
                  if (reference_value < 0 || temp < reference_value) {
                      reference_value = temp;
                      best_split_point = split_point;
                      result.split_sets[0] = sub_set_a;
                      result.split_sets[1] = sub_set_b;
                  }
              }
          }
          //没有分割点，抛出分割异常
          if (result.split_sets[0] == null && result.split_sets[1] == null) throw new SplitException();
          result.split_points = new double[1];
          result.split_points[0] = best_split_point;
      }
      return result;
  }
  
  //计算给定样本集合的熵
  private double entropy(int[] set) {
      HashMap<Double, Integer> counter = new HashMap<Double, Integer>();
      for (int item : set) {
          double label = _labels[item];
          if (counter.get(label) == null) {
              counter.put(label, 1);
          } else {
              int count = counter.get(label) + 1;
              counter.put(label, count);
          }
      }
      
      double result = 0;
      Iterator<Double> iterator = counter.keySet().iterator();
      while (iterator.hasNext()) {
          int count = counter.get(iterator.next());
          double p = (double)count / set.length;
          result += - p * Math.log(p);
      }
      
      return result;
  }
  
  //增益率 C4.5
  private BundleData gain_ratio_use_attribute(int[] set, int attribute) throws SplitException {
      BundleData result = new BundleData();
      double entropy_before_split = entropy(set);
      
      double entropy_after_split = 0;
      double split_information = 0;
      result.split_info = split_with_attribute(set, attribute);
      for (int[] sub_set : result.split_info.split_sets) {
          entropy_after_split += (double)sub_set.length / set.length * entropy(sub_set);
          double p = (double)sub_set.length / set.length;
          split_information += - p * Math.log(p);
      }
      result.floatValue = (entropy_before_split - entropy_after_split) / split_information;
      return result;
  }
  
  private double gain_ratio_use_numerical_attribute(int[] set, int attribute, int[] part_a, int[] part_b) {
      double entropy_before_split = entropy(set);
      double entropy_after_split = (part_a.length * entropy(part_a) + part_b.length * entropy(part_b)) / set.length;
      
      
      double split_information = 0;
      double p = (double)part_a.length / set.length;
      split_information += - p * Math.log(p);
      p = (double)part_b.length / set.length;
      split_information += - p * Math.log(p);
      
      return (entropy_before_split - entropy_after_split) / split_information;
  }
  
  private double mse(int[] set) {
      double mean = mean_value(set);
      
      double temp = 0;
      for (int index : set) {
          double t = _labels[index] - mean;
          temp += t * t;
      }
      return temp / set.length;
  }
  
  private BundleData mse_use_attribute(int[] set, int attribute) throws SplitException {
      BundleData mse_info = new BundleData();
      mse_info.floatValue = 0;
      mse_info.split_info = split_with_attribute(set, attribute);
      for (int[] sub_set : mse_info.split_info.split_sets) {
          mse_info.floatValue += (double)sub_set.length / set.length * mse(sub_set);
      }
      return mse_info;
  }
  
}