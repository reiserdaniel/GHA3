package Package1;

import org.ejml.data.DMatrixIterator;
import org.ejml.simple.SimpleMatrix;

import java.util.concurrent.ThreadLocalRandom;

public class MatrixOperator {
    public static SimpleMatrix ones(int numRows, int numCols){
        SimpleMatrix ones = new SimpleMatrix(numRows, numCols);
        ones = ones.plus(1.0);
        return ones;
    }
    public static SimpleMatrix getMixer(int m){
        SimpleMatrix mixer = new SimpleMatrix(m, m);
        int i = 0;
        while(i < m - 1){
            int randomCol= ThreadLocalRandom.current().nextInt(0, m);
            Boolean right = true;
            for (int j = 0; j < m; j++) {
                if(mixer.get(j,randomCol) == 1){
                    right = false;
                }
            }
            if (right){
                mixer.set(i,randomCol,1.0);
                i++;
            }
        }
        System.out.println("the Mixer is ready");
        return mixer;
    }
    public static SimpleMatrix sigmoid (SimpleMatrix matrix) {
        SimpleMatrix result = matrix;
        SimpleMatrix ones = MatrixOperator.ones(matrix.numRows(), matrix.numCols());
        SimpleMatrix exp = result.negative().elementExp();
        result = ones.elementDiv(ones.plus(exp));
        return result;
    }
    public static SimpleMatrix sigmoidGradient (SimpleMatrix matrix){
        SimpleMatrix ones = MatrixOperator.ones(matrix.numRows(), matrix.numCols());
        return matrix.elementMult(ones.minus(matrix));
    }
    public static SimpleMatrix oneVsAll(SimpleMatrix outputVector){
        SimpleMatrix outputValue = new SimpleMatrix(outputVector.numRows(), 1);
        for (int i = 0; i < outputVector.numRows(); i++) {
            int maxindex= 0;
            for (int j = 0; j < outputVector.numCols(); j++) {
                if(outputVector.get(i,j) > outputVector.get(i,maxindex)){
                    maxindex = j;
                }
            }
            outputValue.set(i,0,maxindex);

            //System.out.println((maxindex+1) + " " + (int) y.get(i,0));
        }
        return outputValue;
    }
    public static SimpleMatrix subMatrix (SimpleMatrix in, int rowFrom, int colFrom, int rowTo, int colTo){
        SimpleMatrix out= new SimpleMatrix(rowTo-rowFrom, colTo - colFrom);
        DMatrixIterator outItr= out.iterator(true,0,0,out.numRows()-1, out.numCols()-1);
        DMatrixIterator inItr = in.iterator(true, rowFrom, colFrom, rowTo-1, colTo-1);

        while(outItr.hasNext() && inItr.hasNext()){
            outItr.next();
            outItr.set(inItr.next());

        }
        return out;
    }
    public static Double cost(DataArray data, NNTrained theta){
        try {
            SimpleMatrix ones = ones(data.getX().numRows(), theta.getK());
            SimpleMatrix prediction= theta.predict(data.getX());
            SimpleMatrix leftSide = data.getY().negative().elementMult(prediction.elementLog());
            SimpleMatrix rightSide = ones.minus(data.getY()).elementMult(ones.minus(prediction).elementLog());
            return leftSide.minus(rightSide).elementSum()/data.getLength();
        } catch (Exception e) {
            e.printStackTrace();
            return null;
        }
    }
}
