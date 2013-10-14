/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package dm13;

import auxiliary.DataSet;
import auxiliary.Evaluation;

/**
 *
 * @author daq
 */
public class TestAss4 {

    public static void main(String[] args) {
        // for RandomForest
        System.out.println("for RandomForest");
        String[] dataPaths = new String[]{"breast-cancer.data", "segment.data"};
        for (String path : dataPaths) {
            DataSet dataset = new DataSet(path);

            // conduct 10-cv 
            Evaluation eva = new Evaluation(dataset, "RandomForest");
            eva.crossValidation();

            // print mean and standard deviation of accuracy
            System.out.println("Dataset:" + path + ", mean and standard deviation of accuracy:" + eva.getAccMean() + "," + eva.getAccStd());
        }

        // for AdaBoost
        System.out.println("\nfor AdaBoost");
        for (String path : dataPaths) {
            DataSet dataset = new DataSet(path);

            // conduct 10-cv 
            Evaluation eva = new Evaluation(dataset, "AdaBoost");
            eva.crossValidation();

            // print mean and standard deviation of accuracy
            System.out.println("Dataset:" + path + ", mean and standard deviation of accuracy:" + eva.getAccMean() + "," + eva.getAccStd());
        }
    }
}
