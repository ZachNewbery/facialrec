package application;

import java.io.ByteArrayInputStream;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import application.FaceRecogniser;

import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfFloat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;
import org.opencv.objdetect.Objdetect;
import org.opencv.videoio.VideoCapture;

import javafx.event.ActionEvent;
import javafx.event.Event;
import javafx.fxml.FXML;
import javafx.scene.control.Button;
import javafx.scene.control.CheckBox;
import javafx.scene.image.Image;
import javafx.scene.image.ImageView;

public class FXController {
    @FXML
    private Button start_btn;
    @FXML
    private ImageView currentFrame;
    @FXML
    private CheckBox graysc_box;
    @FXML
    private ImageView histogram;
    @FXML
    private CheckBox hist_box;
    @FXML
    private CheckBox lbp_box;
    @FXML
    private CheckBox haar_box;

    
    // Required variables.
    private VideoCapture capture = new VideoCapture();
    private ScheduledExecutorService timer;
    private CascadeClassifier faceCascade = new CascadeClassifier();
    private int absoluteFaceSize = 0;
    private CascadeClassifier smileCascade = new CascadeClassifier();
    private int absoluteSmileWidth = 0;
    private int absoluteSmileHeight = 0;
    
    // facerec variables
    private FaceRecogniser rec = new FaceRecogniser("dataset/TrainingData.csv");
    
    private int smileTime = 0;
    
    @FXML
    protected void startCamera(ActionEvent event) {
        if (!this.capture.isOpened()) {
            this.capture.open(0);
            
            Runnable frameGrabber = new Runnable() {
                @Override
                public void run()
                {
                    Mat frame = grabFrame();
                    MatOfByte buffer = new MatOfByte();
                    Imgcodecs.imencode(".png", frame, buffer);
                    Image image = new Image(new ByteArrayInputStream(buffer.toArray()));
                    currentFrame.setImage(image);
                    if(hist_box.isSelected()) {
                        showHist(frame, !graysc_box.isSelected());
                    }
                }
            };
            this.timer = Executors.newSingleThreadScheduledExecutor();
            this.timer.scheduleAtFixedRate(frameGrabber, 0, 33, TimeUnit.MILLISECONDS);
            
            this.start_btn.setText("Stop Camera");
        } else {
            this.capture.release();
            this.start_btn.setText("Start Camera");
        }
    };
    
    @FXML
    protected void selectHaar(Event event) {
        if (this.lbp_box.isSelected()) {
            this.lbp_box.setSelected(false);
        }
        
        this.loadClassifier("resources/haarcascades/haarcascade_frontalface_default.xml");
    }
    
    @FXML
    protected void selectLBP(Event event) {
        if (this.haar_box.isSelected()) {
            this.haar_box.setSelected(false);
        }
        
        this.loadClassifier("resources/lbpcascades/lbpcascade_frontalface_improved.xml");
    }
    
    private void loadClassifier(String path) {
        this.faceCascade.load(path);
        
        this.start_btn.setDisable(false);
    }
    
    private Mat grabFrame() {
        Mat current = new Mat();
        if(this.capture.isOpened()) {
            try {
                this.capture.read(current);
                
                if(!current.empty()) {
                    if(this.haar_box.isSelected() || this.lbp_box.isSelected()) {
                        this.detectFace(current);
                    }
                    if(this.graysc_box.isSelected()) {
                        Imgproc.cvtColor(current, current, Imgproc.COLOR_BGR2GRAY);
                    }
                }
            } catch (Exception e) {
                System.err.println("Exception grabbing image: " + e);
            }
        }
        return current;
    };
    
    private void detectFace(Mat frame) {
        MatOfRect faces = new MatOfRect();
        Mat grayFrame = new Mat();
        
        Imgproc.cvtColor(frame, grayFrame, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(grayFrame, grayFrame);
        
        if (this.absoluteFaceSize == 0)
        {
            int height = grayFrame.rows();
            if (Math.round(height * 0.2f) > 0)
            {
                this.absoluteFaceSize = Math.round(height * 0.2f);
            }
        }
        
        this.faceCascade.detectMultiScale(grayFrame, faces, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE,
                new Size(this.absoluteFaceSize, this.absoluteFaceSize), new Size());
        
        
        Rect[] facesArray = faces.toArray();
        for (int i = 0; i < facesArray.length; i++) {
            if(detectSmile(frame, facesArray[i])) {
                this.smileTime++;
                if(smileTime >= 60) {
                    Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 255, 0), 3);
                    double[] results = rec.recogniseFace(grayFrame.submat(facesArray[i]));
                    Imgproc.putText(frame, results[0] + ": " + results[1] + "%", new Point(facesArray[i].x, facesArray[i].y - 10), 1, 1.3, new Scalar(0, 255, 0));
                } else {
                    Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(255, 0, 0), 3);
                }               
            } else {
                Imgproc.rectangle(frame, facesArray[i].tl(), facesArray[i].br(), new Scalar(0, 0, 255), 3);
                this.smileTime = 0;
            }
        }
    }
    
    private boolean detectSmile(Mat full, Rect face) {
        this.smileCascade.load("resources/haarcascades/haarcascade_smile.xml");
        MatOfRect smiles = new MatOfRect();
        face.height = face.height + 20;
        Mat faceROI = full.submat(face);
        
        if(this.absoluteSmileWidth == 0) {
            int width = faceROI.cols();
            if (Math.round(width * 0.75f) > 0)
            {
                this.absoluteSmileWidth = Math.round(width * 0.75f);
            }
        }
        
        if(this.absoluteSmileHeight == 0) {
            int height = faceROI.rows();
            if (Math.round(height * 0.25f) > 0) {
                this.absoluteSmileHeight = Math.round(height * 0.25f);
            }
        }
        
        // call detectMultiScale to find a smile
        this.smileCascade.detectMultiScale(faceROI, smiles, 1.1, 2, 0 | Objdetect.CASCADE_SCALE_IMAGE, new Size(this.absoluteSmileWidth, this.absoluteSmileHeight), new Size());
        
        Rect[] smileArray = smiles.toArray();
        
        // check each smile, and verify position w.r.t face rectangle
        for(int i = 0; i < smileArray.length; i++) {
            Rect smile = smileArray[i];
            if((smile.y > (face.y * 0.4f))) {
                Imgproc.rectangle(faceROI, smileArray[i].tl(), smileArray[i].br(), new Scalar(0, 255, 0), 3);
                return true;
            }
        }
        return false;
    }
    
    public void showHist(Mat frame, boolean color) {
        List<Mat> layers = new ArrayList<Mat>();
        Core.split(frame, layers);
        
        MatOfInt histSize = new MatOfInt(256);
        MatOfInt channels = new MatOfInt(0);
        MatOfFloat histRange = new MatOfFloat(0, 256);
        
        Mat hist_r = new Mat();
        Mat hist_b = new Mat();
        Mat hist_g = new Mat();
        
        Imgproc.calcHist(layers.subList(0, 1), channels, new Mat(), hist_b, histSize, histRange, false);
        
        if(color) {
            Imgproc.calcHist(layers.subList(1, 2), channels, new Mat(), hist_g, histSize, histRange, false);
            Imgproc.calcHist(layers.subList(2, 3), channels, new Mat(), hist_r, histSize, histRange, false);
        }
        
        int width = 250;
        int height = 150;
        int bin_width = (int) Math.round(width / histSize.get(0, 0)[0]);
        
        
        Mat histImage = new Mat(height, width, CvType.CV_8UC3, new Scalar(0, 0, 0));
        
        Core.normalize(hist_b, hist_b, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
        if(color) {
            Core.normalize(hist_g, hist_g, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
            Core.normalize(hist_r, hist_r, 0, histImage.rows(), Core.NORM_MINMAX, -1, new Mat());
        }
        
        for(int i = 1; i < histSize.get(0, 0)[0]; i++) {
            Imgproc.line(histImage, new Point(bin_width * (i - 1), height - Math.round(hist_b.get(i - 1, 0)[0])),
                    new Point(bin_width * (i), height - Math.round(hist_b.get(i, 0)[0])), new Scalar(255, 0, 0), 2, 8, 0);
            
            if(color) {
                Imgproc.line(histImage, new Point(bin_width * (i - 1), height - Math.round(hist_b.get(i - 1, 0)[0])),
                        new Point(bin_width * (i), height - Math.round(hist_g.get(i, 0)[0])), new Scalar(0, 255, 0), 2, 8, 0);
                Imgproc.line(histImage, new Point(bin_width * (i - 1), height - Math.round(hist_b.get(i - 1, 0)[0])),
                        new Point(bin_width * (i), height - Math.round(hist_r.get(i, 0)[0])), new Scalar(0, 0, 255), 2, 8, 0);
            }
        }
       
        MatOfByte buffer = new MatOfByte();
        Imgcodecs.imencode(".png", histImage, buffer);
        Image image = new Image(new ByteArrayInputStream(buffer.toArray()));
        
        histogram.setImage(image);

    }
}
