package Package1;

import org.ejml.simple.SimpleMatrix;

import java.io.Serializable;
import java.util.ArrayList;
import static Package1.MatrixOperator.*;

public class NNTrained implements Serializable{
    ArrayList<SimpleMatrix> thetaArray;
    NNTrained(ArrayList<SimpleMatrix> thetaArray){
        this.thetaArray = thetaArray;
    }

    public SimpleMatrix predict (SimpleMatrix X) throws Exception{
        SimpleMatrix a = X;
        for(SimpleMatrix theta : thetaArray){
            a = ones(a.numRows(),1).combine(0,1, a);
            a = a.mult(theta);
            a = sigmoid(a);
        }
        return a;
    }

    public Integer getK(){
        SimpleMatrix theta = thetaArray.get(thetaArray.size()-1);
        return theta.numCols();
    }

    public ArrayList<SimpleMatrix> getThetaArray() {
        return thetaArray;
    }
}
