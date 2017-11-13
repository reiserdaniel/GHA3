package Package1;



import org.ejml.simple.SimpleMatrix;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Random;


public class NNTester {

    public static void main(String[] args) {
        //test1(); //playground to try SimpleMatrix class, basic constructor for NNBase class
        //test2(); //import already trained network, test functions: rateNetwork1vsall, sigmoid, ones, oneVsAll, predict
        //test3(); //testing the backpropagation, and the learning abilities of the network class
        test4();
    }
    private static void test1(){
        SimpleMatrix a = new SimpleMatrix(4,4);
        ArrayList<Integer> struct = new ArrayList<>();
        struct.add(10);
        struct.add(4);
        struct.add(4);
        Random rand = new Random();
        SimpleMatrix b = SimpleMatrix.random64(5,4,-1.0,1.0, rand);
        a = SimpleMatrix.random64(5,5,-1.0,1.0, rand);
        a.print();
        b.print();
        a.combine(0,2,b).print();

        NNBase.subMatrix(a,0,1,5,5).print();
        return;
    }

    private static void test2(){
        ArrayList<SimpleMatrix> thetaArray = new ArrayList<>();
        try {
            SimpleMatrix X = SimpleMatrix.loadBinary("Xser");
            SimpleMatrix yUnrolled = SimpleMatrix.loadBinary("yser");
            thetaArray.add(SimpleMatrix.loadBinary("theta1ser").transpose());
            thetaArray.add(SimpleMatrix.loadBinary("theta2ser").transpose());
            SimpleMatrix y = new SimpleMatrix(yUnrolled.numRows(), 10);
            for (int i = 0; i < yUnrolled.numRows(); i++) {
                for (int j = 0; j < 10; j++) {
                    if(yUnrolled.get(i,0) == j+1){
                        y.set(i,j,1.0);
                    }
                }
            }
            NNBase test = new NNBase(thetaArray, X, y, 0.001);
            //test.printNetwork();

            System.out.println(test.cost(test.X,test.y) + " " + test.costWithReg(test.X,test.y));
            test.rateNetwork1vsAll();
            //System.out.println(test.gradientChecking());
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void test3(){

        try {
            SimpleMatrix X = SimpleMatrix.loadBinary("Xser");
            SimpleMatrix yUnrolled = SimpleMatrix.loadBinary("yser");
            SimpleMatrix y = new SimpleMatrix(yUnrolled.numRows(), 10);
            for (int i = 0; i < yUnrolled.numRows(); i++) {
                for (int j = 0; j < 10; j++) {
                    if(yUnrolled.get(i,0) == j+1){
                        y.set(i,j,1.0);
                    }
                }
            }
            ArrayList<Integer> structure = new ArrayList<>();
            structure.add(50);
            structure.add(25);
            structure.add(20);
            structure.add(15);
            NNBase test = new NNBase(structure, X, y);
            ArrayList<Double> lambdaList = new ArrayList<>();
            lambdaList.add(0.001);
            lambdaList.add(0.003);
            for (int i = 0; i < 4; i++) {
                lambdaList.add(lambdaList.get(lambdaList.size()-2)*10);
                lambdaList.add(lambdaList.get(lambdaList.size()-2)*10);
            }
            test.getBestLambda(lambdaList);
            test.gradientDescent(800,1.0);
            test.costWithReg(test.XTest,test.yTest);
            test.rateNetwork1vsAll();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static void test4(){
        ImageReader testImage = new ImageReader(new File("/Users/daniel/Documents/IdeaProjects/Grosshausaufgabe/data/faces/0000227285.jpeg"));
        int[] testImageArr = testImage.img.getRGB(0,0,3, 3, null,0,1);
        for (int i :
                testImageArr) {
            System.out.print(i);
        }
    }
}

