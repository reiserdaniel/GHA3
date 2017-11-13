package Package1;

import java.io.File;

public class DataSerializer {
    private File[] dirArr; //directories where the data are.
    private int bSize;
    private int K;
    private Double[] relSize; //relative sizes of the directories
    private File outData; //mappa sorbatett serializalt data elemekkel
    DataSerializer(File[] directories, int batchSize){
        dirArr = directories;
        bSize = batchSize;
        K = dirArr.length;
        relSize = new Double[K];
        double sum = 0;
        for (File dir:
                dirArr) {
            sum+= (double) dir.listFiles().length;
        }
        for (int i = 0; i < K; i++) {
            relSize[i] = (double) dirArr[i].listFiles().length/sum;
        }

        for (int i = 0; i < sum/(K*bSize)-1; i++) {
            for (int j = 0; j < K; j++) {
                //TODO beolvasni K mappa'bol relsize*batchsize elemet, es elmenteni
            }
        }
    }
}
