package Package1;

import org.ejml.simple.SimpleMatrix;
import java.io.Serializable;

public class DataArray implements Serializable{
    private SimpleMatrix X;
    private SimpleMatrix y;

    DataArray(SimpleMatrix X, SimpleMatrix y) throws Exception {
        this.X = X;
        this.y = y;
        if (X.numRows() != y.numRows()){
            Exception e = new Exception("Dimensions of the matrices does not fit");
            throw e;
        }
    }

    public SimpleMatrix getX() {
        return X;
    }

    public SimpleMatrix getY() {
        return y;
    }

    public int getLength(){
        return X.numRows();
    }
}
