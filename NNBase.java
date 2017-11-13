package Package1;

import java.util.ArrayList;
import java.util.Collections;
import java.util.Iterator;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.ejml.data.DMatrixIterator;
import org.ejml.simple.SimpleMatrix;

import static Package1.MatrixOperator.*;

public class NNBase {
    private ArrayList<Integer> structure; //structure of the network, does not contains bias
    private ArrayList<SimpleMatrix> thetaArray; //the network
    private ArrayList<SimpleMatrix> aArray; //a matrices made during predict()
    private ArrayList<SimpleMatrix> zArray; //y matrices made during predict()
    private Integer n; //input vector size = sturcture(0)
    private Integer K; //output vector size = structure.last
    private Integer m; //dataset size
    private Integer trainDataSize; //% of X
    private Integer crossValidationDataSize; // % of X
    private Integer testDataSize; // % of X, train + cross + test = 1;
    private Double lambda; //regularisation value
    private SimpleMatrix X; //dataset  m*n size
    private SimpleMatrix y; //output value m*K size
    private SimpleMatrix XCross;
    private SimpleMatrix yCross;
    private SimpleMatrix XTest;
    private SimpleMatrix yTest;


    NNBase(ArrayList<Integer> structure, SimpleMatrix X, SimpleMatrix y) {
        trainDataSize = X.numRows()*6/10;
        crossValidationDataSize = trainDataSize/3;
        testDataSize = X.numRows()-trainDataSize-crossValidationDataSize;
        K = y.numCols();
        m = trainDataSize;
        n = X.numCols();
        lambda = null;
        SimpleMatrix mixer = getMixer(X.numRows());
        this.X = subMatrix(mixer.mult(X),0,0,trainDataSize,X.numCols());
        this.y = subMatrix(mixer.mult(y),0,0,trainDataSize,K);
        this.XCross = subMatrix(mixer.mult(X),trainDataSize,0,trainDataSize + crossValidationDataSize,X.numCols());
        this.yCross = subMatrix(mixer.mult(y),trainDataSize,0,trainDataSize + crossValidationDataSize,K);
        this.XTest = subMatrix(mixer.mult(X),trainDataSize + crossValidationDataSize,0,X.numRows(),X.numCols());
        this.yTest = subMatrix(mixer.mult(y),trainDataSize + crossValidationDataSize,0,y.numRows(),K);

        this.structure = new ArrayList<>();
        this.structure.add(n);
        for (int a : structure) {
            this.structure.add(a);
        }
        this.structure.add(K);

        this.thetaArray = new ArrayList<>();
        Random rand = new Random();
        for (int i = 1; i<this.structure.size(); i++) {
            this.thetaArray.add(SimpleMatrix.random64(this.structure.get(i-1)+1, this.structure.get(i), -0.1,  0.1,  rand));
        }
    }

    NNBase(ArrayList<SimpleMatrix> thetaArray, SimpleMatrix X, SimpleMatrix y, Double lambda){
        this.thetaArray = thetaArray;
        this.X = X;
        this.y = y;
        K = y.numCols();
        m = X.numRows();
        n = X.numCols();
        this.lambda = lambda;
        //structure.add(thetaArray.get(0).numCols() -1);
        //for (SimpleMatrix theta: thetaArray) {
        //    structure.add(theta.numCols());
        //}

    }

    public void printNetwork(){
        for(SimpleMatrix theta : thetaArray){
            theta.print();
        }
    }


    public  SimpleMatrix predict (SimpleMatrix X) {
        SimpleMatrix a = X;
        zArray = new ArrayList<>();
        aArray = new ArrayList<>();
        for(SimpleMatrix theta : thetaArray){
            zArray.add(a);
            a = ones(a.numRows(),1).combine(0,1, a);
            aArray.add(a);
            a = a.mult(theta);

            a = sigmoid(a);

        }
        return a;
    }


    public Double cost(SimpleMatrix X, SimpleMatrix y){
        SimpleMatrix ones = ones(X.numRows(), K);
        SimpleMatrix prediction = predict(X);
        SimpleMatrix leftSide = y.negative().elementMult(prediction.elementLog());
        SimpleMatrix rightSide = ones.minus(y).elementMult(ones.minus(prediction).elementLog());
        return leftSide.minus(rightSide).elementSum()/m;
    }

    public Double costWithReg(SimpleMatrix X, SimpleMatrix y){
        Double regularisation =0.0;
        Double cost = cost(X,y);
        if(!isRegularised()){
            return cost;
        }
        for(SimpleMatrix theta : thetaArray){
            DMatrixIterator itr = theta.iterator(true,0,0,theta.numRows()-1,theta.numCols()-1);
            while(itr.hasNext()){
                Double a = itr.next();
                regularisation += a*a;
            }
        }
        return regularisation*lambda/(2*m) + cost;
    }

    public void rateNetwork1vsAll(){
        SimpleMatrix prediction = oneVsAll(predict(XTest));
        int positiv = 0;
        int negativ = 0;
        for (int i = 0; i < prediction.numRows(); i++) {
            int yRolled = 0;
            for (int j = 0; j < K; j++) {
                if (yTest.get(i,j) == 1){
                    yRolled = j;
                }
            }
            if (prediction.get(i,0)  == yRolled){
                positiv++;
            }
            else{
                negativ++;
            }
            //System.out.println((maxindex+1) + " " + (int) y.get(i,0));
        }
        System.out.println("There was " + positiv + " right guess, and " + negativ + " wrong guess. Thats " + (double)positiv/(double)(positiv+negativ)*100 + "%.");
    }

    public ArrayList<SimpleMatrix> backPropagation() {
        ArrayList<SimpleMatrix> deltaArray = new ArrayList<>();
        ArrayList<SimpleMatrix> thetaGradArray = new ArrayList<>();
        deltaArray.add(predict(X).minus(y));
        for (int i = 1; i < thetaArray.size(); i++) {
            SimpleMatrix theta = thetaArray.get(thetaArray.size() - i).transpose();
            SimpleMatrix delta = deltaArray.get(deltaArray.size() - 1).mult(theta);
            delta = subMatrix(delta, 0, 1, delta.numRows(), delta.numCols());
            delta = delta.elementMult(sigmoidGradient(zArray.get(zArray.size() - i)));
            deltaArray.add(delta);

        }
        for (int i = 0; i < thetaArray.size(); i++) {
            SimpleMatrix thetaGrad = deltaArray.get(i).transpose();
            thetaGrad = thetaGrad.mult(aArray.get(aArray.size()-i-1)).transpose();
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

    public Boolean gradientChecking(){
        ArrayList<SimpleMatrix> thetaGradArray = backPropagation();
        //ArrayList<SimpleMatrix> thetaArray = new ArrayList<>(this.thetaArray);
        Iterator TGAitr = thetaGradArray.iterator();
        Iterator TAitr = thetaArray.iterator();
        Boolean reValue = true;
        Double epsilon = 0.00001;

        while (TGAitr.hasNext() && TAitr.hasNext()){
            //System.out.println(TGAitr.hasNext() + " " + TAitr.hasNext());
            SimpleMatrix gradient = (SimpleMatrix) TGAitr.next();
            //DMatrixIterator gradientItr =  gradient.iterator(true, 0,0,gradient.numRows()-1,gradient.numCols()-1);
            SimpleMatrix theta = (SimpleMatrix) TAitr.next();
            //DMatrixIterator thetaItr = theta.iterator(true, 0,0, theta.numRows()-1,theta.numCols()-1);
            for (int i = 0; i < 10; i++) {
                int randomRow= ThreadLocalRandom.current().nextInt(0, theta.numRows());
                int randomCol = ThreadLocalRandom.current().nextInt(0, theta.numCols());

                Double value = theta.get(randomRow,randomCol);
                theta.set(randomRow,randomCol,value - epsilon);
                Double loss1 = costWithReg(X,y);
                theta.set(randomRow,randomCol,value + epsilon);
                Double loss2 = costWithReg(X,y);
                theta.set(randomRow,randomCol,value );

                //System.out.print(".");

                Double gradientValue = (loss2 - loss1)/(2*epsilon);
                if (gradientValue - gradient.get(randomRow,randomCol) > 0.001 ||
                        gradientValue - gradient.get(randomRow,randomCol) < -0.001){
                    return false;
                }
                System.out.println(gradientValue+ "\t" +  gradient.get(randomRow,randomCol));
            }
            System.out.println();
        }
        return reValue;
    }

    private void gradientDescent(int iterNum, Double alpha){
        for (int i = 0; i < iterNum; i++) {
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
            if(i%30 == 0) System.out.println("iteration: " + i + "\tTraindata cost: " + cost(X,y) + "\tTestdata cost: " + cost(XTest,yTest)*trainDataSize/testDataSize);
        }
    }

    public Double getBestLambda(ArrayList<Double> lambdaList){
        ArrayList<Double> costArray = new ArrayList<>();
        this.lambda = lambdaList.get(0);
        for (Double lambda: lambdaList) {
            this.lambda = lambda;
            gradientDescent(200,3.0);
            Double cost = cost(XCross,yCross);
            costArray.add(cost);
            System.out.println("Lambda = " + lambda + "\tCost: " + cost);
        }
        int j = 0;
        for (int i = 0; i<costArray.size();i++) {
            if (costArray.get(i)  < costArray.get(j)){
                j = i;
            }
        }
        this.lambda = lambdaList.get(j);
        return lambda;
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
