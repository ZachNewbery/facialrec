package application;

import java.io.ByteArrayInputStream;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.concurrent.TimeUnit;
import application.FaceRecogniser;

import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfRect;
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
                    System.out.println("Detected Face: " + rec.recogniseFace(frame.submat(facesArray[i])));
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
}
