package application;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.face.EigenFaceRecognizer;

public class FaceRecogniser {
    private String trainingPath;
    private ArrayList<Mat> images = new ArrayList<>();
    private ArrayList<Integer> labels = new ArrayList<>();
    private EigenFaceRecognizer model;
    
    FaceRecogniser(String path) {
        this.trainingPath = path;
        
        // populate arraylists using training data.
        System.out.println("Populating training set...");
        populateLists(this.trainingPath, this.labels, this.images);
        System.out.println("Done population, training model...");
        
        // obtaining sample data
        Mat sample = images.get(images.size() - 1);
        Integer sampleLabel = this.labels.get(this.labels.size() - 1);
        MatOfInt labelsMat = new MatOfInt();
        labelsMat.fromList(this.labels);
        this.model = EigenFaceRecognizer.create();
        
        // training model
        this.model.train(this.images, labelsMat);
        
        // testing model
        int[] testLabels = new int[1];
        double[] confidence = new double[1];
        System.out.println("Testing trained model...");
        this.model.predict(sample, testLabels, confidence);
        
        System.out.println("***Predicted Label: " + testLabels[0] + "***");
        System.out.println("***Actual Label: " + sampleLabel + "***");
        System.out.println("***Confidence: " + confidence[0] + "***");
    }

    private void populateLists(String trainingPath, ArrayList<Integer> labels, ArrayList<Mat> images) {
        BufferedReader br;
        
        try {
            br = new BufferedReader(new FileReader(trainingPath));
            String line;
            while((line=br.readLine()) != null) {
                String[] tokens = line.split("\\;");
                Mat readImg = Imgcodecs.imread(tokens[0], 0);
                images.add(readImg);
                labels.add(Integer.parseInt(tokens[1]));
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (NumberFormatException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
    
    public double[] recogniseFace(Mat face) {
        double[] results = new double[2];
        Mat comp = new Mat();
        int[] label = new int[1];
        double[] conf = new double[1];
        Size sz = new Size(112, 92);
        Imgproc.resize(face, comp, sz);
        this.model.predict(comp, label, conf);
        
        results[0] = label[0];
        results[1] = convertConf(conf[0]);
        return results;
    }
    
    public static double convertConf(double conf) {
        double percent = (10000 - conf)/100;
        return percent;
    }
}
