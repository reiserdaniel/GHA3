package Package1;

import java.io.*;
import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.ejml.data.DMatrixIterator;
import org.ejml.simple.SimpleMatrix;

import static Package1.MatrixOperator.*;

public class NNBatch {
    private ArrayList<Integer> structure; //structure of the network, does not contains bias
    private ArrayList<SimpleMatrix> thetaArray; //the network
    private ArrayList<SimpleMatrix> zArray; //y matrices made during predict()
    private Integer n; //input vector size = sturcture(0)
    private Integer K; //output vector size = structure.last
    private Integer b; //batch size
    private Integer m; //whole dataset size
    private Integer bNum; //number of batches
    private Double lambda; //regularisation value
    private DataArray data;
    private File[] batches;
    private Integer thisBatch;


    private SimpleMatrix predict (SimpleMatrix X) {
        SimpleMatrix a = X;
        zArray = new ArrayList<>();
        for(SimpleMatrix theta : thetaArray){
            zArray.add(a);
            a = ones(a.numRows(),1).combine(0,1, a);
            a = a.mult(theta);
            a = sigmoid(a);

        }
        return a;
    }

    private void nextBatch(){
        if(thisBatch < bNum) {
            thisBatch++;
        }
        else{
            thisBatch = 0;
        }
        try {
            InputStream is = new FileInputStream(batches[thisBatch]);
            InputStream buffer = new BufferedInputStream(is);
            ObjectInput input = new ObjectInputStream(buffer);
            data = (DataArray) input.readObject();
        }
        catch (Exception e){
            e.printStackTrace();
        }
    }

    private ArrayList<SimpleMatrix> backPropagation() {
        ArrayList<SimpleMatrix> deltaArray = new ArrayList<>();
        ArrayList<SimpleMatrix> thetaGradArray = new ArrayList<>();
        deltaArray.add(predict(data.getX()).minus(data.getY()));
        for (int i = 1; i < thetaArray.size(); i++) {
            SimpleMatrix theta = thetaArray.get(thetaArray.size() - i).transpose();
            SimpleMatrix delta = deltaArray.get(deltaArray.size() - 1).mult(theta);
            delta = subMatrix(delta, 0, 1, delta.numRows(), delta.numCols());
            delta = delta.elementMult(sigmoidGradient(zArray.get(zArray.size() - i)));
            deltaArray.add(delta);

        }
        for (int i = 0; i < thetaArray.size(); i++) {
            SimpleMatrix thetaGrad = deltaArray.get(i).transpose();
            SimpleMatrix a = zArray.get(zArray.size()-i-1);
            a = ones(a.numRows(),1).combine(0,1, a);
            thetaGrad = thetaGrad.mult(a).transpose();
            thetaGrad = thetaGrad.divide((double) m);

            //regularization
            if(isRegularised()){
                SimpleMatrix regularizationMask = new SimpleMatrix(thetaGrad.numRows(), 1);
                regularizationMask = regularizationMask.combine(
                        0,
                        1,
                        ones(thetaGrad.numRows(), thetaGrad.numCols()-1));

                thetaGrad = thetaGrad.plus(regularizationMask.elementMult(thetaArray.get(thetaArray.size() - i-1).transpose().divide(m/lambda).transpose()));
            }
            thetaGradArray.add(thetaGrad);
        }
        Collections.reverse(thetaGradArray);
        return thetaGradArray;

    }

    private void gradientDescent(Boolean iter, Double alpha){
        int i = 0;
        while(iter){
            nextBatch();
            ArrayList<SimpleMatrix> thetaGrad = backPropagation();
            Iterator thetaGradArrayItr = thetaGrad.iterator();
            Iterator thetaArrayItr = thetaArray.iterator();
            while(thetaArrayItr.hasNext()&& thetaGradArrayItr.hasNext()){
                SimpleMatrix thetaGradient = (SimpleMatrix) thetaGradArrayItr.next();
                DMatrixIterator thetaGradItr = thetaGradient.iterator(true,0,0,thetaGradient.numRows()-1,thetaGradient.numCols()-1);
                SimpleMatrix theta = (SimpleMatrix) thetaArrayItr.next();
                DMatrixIterator thetaItr = theta.iterator(true,0,0,theta.numRows()-1,theta.numCols()-1);
                while(thetaItr.hasNext()&& thetaGradItr.hasNext()){
                    Double t = thetaItr.next();
                    thetaItr.set(t-alpha*thetaGradItr.next());
                }
            }
            if(i%30 == 0) System.out.println("iteration: " + i + "\tTraindata cost: " + cost(data, new NNTrained(thetaArray)));
        }
    }

    public void setLambda(Double lambda) {
        this.lambda = lambda;
    }

    private Boolean isRegularised(){
        if(lambda == null){
            return false;
        }
        if(lambda == 0.0){
            return false;
        }
        return true;
    }

    public ArrayList<Integer> getStructure() {
        return structure;
    }

    public ArrayList<SimpleMatrix> getThetaArray() {
        return thetaArray;
    }
}
