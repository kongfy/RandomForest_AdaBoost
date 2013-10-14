package auxiliary;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.logging.Level;
import java.util.logging.Logger;

/**
 *
 * @author daq
 */
public class DataSet {

    private boolean[] isCategory;
    private double[][] features;
    private double[] labels;
    private int numAttributes;
    private int numInstnaces;

    public DataSet(String path) {
        try {
            BufferedReader reader = new BufferedReader(new FileReader(path));
            String[] attInfo = reader.readLine().split(","); // attributes info
            numAttributes = attInfo.length - 1;
            isCategory = new boolean[numAttributes + 1];
            for (int i = 0; i < isCategory.length; i++) {
                isCategory[i] = Integer.parseInt(attInfo[i]) == 1 ? true : false;
            }

            numInstnaces = 0;
            while (reader.readLine() != null) {
                numInstnaces++;
            }

            features = new double[numInstnaces][numAttributes];
            labels = new double[numInstnaces];
            System.out.println("reading " + numInstnaces + " exmaples with " + numAttributes + " attributes");

            reader = new BufferedReader(new FileReader(path));
            reader.readLine();
            String line;
            int ind = 0;
            while ((line = reader.readLine()) != null) {
                String[] atts = line.split(",");
                for (int i = 0; i < atts.length - 1; i++) {
                    features[ind][i] = Double.parseDouble(atts[i]);
                }
                labels[ind] = Double.parseDouble(atts[atts.length - 1]);
                ind++;
            }
            reader.close();
        } catch (FileNotFoundException ex) {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, null, ex);
        } catch (IOException ex) {
            Logger.getLogger(DataSet.class.getName()).log(Level.SEVERE, null, ex);
        }
    }

    public boolean[] getIsCategory() {
        return isCategory;
    }

    public double[][] getFeatures() {
        return features;
    }

    public double[] getLabels() {
        return labels;
    }

    public int getNumAttributes() {
        return numAttributes;
    }

    public int getNumInstnaces() {
        return numInstnaces;
    }
}
