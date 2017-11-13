package Package1;

import org.ejml.simple.SimpleMatrix;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.util.ArrayList;
import java.util.Collections;

public class MatrixReader {
    SimpleMatrix out;
    public MatrixReader(File in, String outFileName){
        try {

            FileReader fr = new FileReader(in);
            BufferedReader br = new BufferedReader(fr);
            String line;
            ArrayList<double[]> matrixInList = new ArrayList<>();
            line = br.readLine();
            while(line != null){
                //System.out.println(line);
                String[] splittedLines = line.split(",");
                double[] doubleLine = new double[splittedLines.length];
                for(int i = 0; i<splittedLines.length; i++){
                    doubleLine[i]= Double.parseDouble(splittedLines[i]);
                }
                matrixInList.add(doubleLine);
                line = br.readLine();
            }
            double[][] arrayArrayMatrix = new double[matrixInList.size()][matrixInList.get(1).length];
            //arrayArrayMatrix = (double[][]) matrixInList.toArray();
            //TODO for ciklussal  egyenként kéne átmásolni a lista elemeit egy tömbbe
            for (int i = 0; i<matrixInList.size(); i++){
                arrayArrayMatrix[i] = matrixInList.get(i);
            }

            out = new SimpleMatrix(arrayArrayMatrix);
            //out.print();
            out.saveToFileBinary(outFileName);
        }catch( Exception e){
            e.printStackTrace();
        }

    }

}
