/*
 * To change this template, choose Tools | Templates
 * and open the template in the editor.
 */
package auxiliary;

import java.io.Serializable;

/**
 *
 * @author daq
 */
public abstract class Classifier implements Cloneable, Serializable {

    public abstract void train(boolean[] isCategory, double[][] features, double[] labels);

    public abstract double predict(double[] features);
}
