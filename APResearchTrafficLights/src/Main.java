import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.lang.reflect.Array;
import java.util.ArrayList;
import java.util.Scanner;

//import org.opencv.*;
import org.opencv.core.*;
import org.opencv.core.Point;
import org.opencv.features2d.FastFeatureDetector;
import org.opencv.features2d.Features2d;
import org.opencv.features2d.SimpleBlobDetector;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;

public class Main {
    static PrintWriter writer;
    static Scanner scanner;
    static ArrayList<String> key;
    static{
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }
    public static BufferedImage Mat2BufferedImage(Mat m) {
        int type = BufferedImage.TYPE_BYTE_GRAY;
        if (m.channels() > 1) {
            type = BufferedImage.TYPE_3BYTE_BGR;
        }
        int bufferSize = m.channels() * m.cols() * m.rows();
        byte[] b = new byte[bufferSize];
        m.get(0, 0, b);
        BufferedImage image = new BufferedImage(m.cols(), m.rows(), type);
        final byte[] targetPixels = ((DataBufferByte) image.getRaster().getDataBuffer()).getData();
        System.arraycopy(b, 0, targetPixels, 0, b.length);
        return image;
    }
    public static void displayImage(Image img){
        ImageIcon icon = new ImageIcon(img);
        JFrame frame = new JFrame();
        frame.setLayout(new FlowLayout());
        frame.setSize(img.getWidth(null)+50, img.getHeight(null)+50);
        JLabel lbl = new JLabel();
        lbl.setIcon(icon);
        frame.add(lbl);
        frame.setVisible(true);
        frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
    }


    public static void main(String[] args) {

        /*
         * VALUE BANK
         * RGB:
         *   RED { [200,255], [0,100], [0,50] }
         *   YELLOW { [200,255] [200,255] [0,100] }
         *   GREEN { [0,70] [165, 255] [70,150] }
         * HSV:
         *   RED { [0,10] [215,255] [110,255] }
         *   YELLOW { [20,45] [200,255] [242,255] }
         *   GREEN { [60,100] [215,255] [110,255] }
         * YCbCr:
         *   RED
         *   YELLOW
         *   GREEN
         * */
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        double[][][] rgb = {
                             { {200,255}, {0,100}, {0,50} },
                             { {200,255}, {200,255}, {0,100} },
                             { {0,70}, {165,255}, {70,150} }
                           };
        double[][][] hsv = {
                                { {0,10}, {225,255}, {250,255} },
                                { {20,45}, {200,255}, {242,255} },
                                { {60,100}, {215,255}, {110,255} }
                           };
        double[][][] ycrcb = {
                                { {80,125}, {245,255}, {80,100} },
                                { {200,255}, {100,200}, {0,50} },
                                { {150,255}, {25,125}, {25,125} }
                             };
        key = new ArrayList<String>();
        System.out.println("Welcome to the OpenCV Traffic Light Detection Software");
        int files = 90;
        try {
            // randomizing order of the worksheets
            writer = new PrintWriter("output.txt", "UTF-8");
            File keyFile = new File("key.txt");
            scanner = new Scanner(keyFile);
            for( int i = 1; i <= files ; i++ ){
                String line = scanner.nextLine();
                System.out.println( line );
                key.add( line );
            }
            System.out.println( key.size() );
            writer.println( "Files to be Scanned: " + files );
            writer.println( "Starting RGB" );
            search( rgb, files,0 );
            writer.println("");
            writer.println("Starting HSV");
            search( hsv, files, 1 );
            writer.println("");
            writer.println("Starting YCbCr");
            search( ycrcb, files, 2 );
            //displayImage(Mat2BufferedImage(out));
            writer.close();
            System.out.println( "Completed Function. AP Research = Done." );
        }catch (Exception e){
            // bnsystem.out.println(e);
            System.out.println(e);
            writer.println( "There was a Error, this dataset is INVALID" );
        }

    }
    /**
     * Segment an image based on hue, saturation, and value ranges.
     *
     * @param input The image on which to perform the HSL threshold.
     * @param hue   The min and max hue
     * @param sat   The min and max saturation
     * @param val   The min and max value
     * @param out   The image in which to store the output.
     */
    private static void threshold(Mat input, double[] hue, double[] sat, double[] val, Mat out)
    {
        //Imgproc.cvtColor(input, out, Imgproc.COLOR_BGR2HSV);
        Core.inRange( input, new Scalar(hue[0], sat[0], val[0]),
                new Scalar(hue[1], sat[1], val[1]), out);
    }

    /**
     * Expands area of higher value in an image.
     * @param src the Image to dilate.
     * @param kernel the kernel for dilation.
     * @param anchor the center of the kernel.
     * @param iterations the number of times to perform the dilation.
     * @param borderType pixel extrapolation method.
     * @param borderValue value to be used for a constant border.
     * @param dst Output Image.
     */
    private static void cvDilate(Mat src, Mat kernel, Point anchor, double iterations,
                          int borderType, Scalar borderValue, Mat dst) {
        if (kernel == null) {
            kernel = new Mat();
        }
        if (anchor == null) {
            anchor = new Point(-1,-1);
        }
        if (borderValue == null){
            borderValue = new Scalar(-1);
        }
        Imgproc.dilate(src, dst, kernel, anchor, (int)iterations, borderType, borderValue);
    }

    /**
     * Detects groups of pixels in an image.
     * @param input The image on which to perform the find blobs.
     * @param minArea The minimum size of a blob that will be found
     * @param circularity The minimum and maximum circularity of blobs that will be found
     * @param darkBlobs The boolean that determines if light or dark blobs are found.
     * @param blobList The output where the MatOfKeyPoint is stored.
     */
    private static void findBlobs(Mat input, double minArea, double[] circularity,
                           Boolean darkBlobs, MatOfKeyPoint blobList) {
        SimpleBlobDetector blobDet = SimpleBlobDetector.create();
        try {
            File tempFile = File.createTempFile("config", ".xml");

            StringBuilder config = new StringBuilder();

            config.append("<?xml version=\"1.0\"?>\n");
            config.append("<opencv_storage>\n");
            config.append("<thresholdStep>10.</thresholdStep>\n");
            config.append("<minThreshold>50.</minThreshold>\n");
            config.append("<maxThreshold>220.</maxThreshold>\n");
            config.append("<minRepeatability>2</minRepeatability>\n");
            config.append("<minDistBetweenBlobs>10.</minDistBetweenBlobs>\n");
            config.append("<filterByColor>1</filterByColor>\n");
            config.append("<blobColor>");
            config.append((darkBlobs ? 0 : 255));
            config.append("</blobColor>\n");
            config.append("<filterByArea>1</filterByArea>\n");
            config.append("<minArea>");
            config.append(minArea);
            config.append("</minArea>\n");
            config.append("<maxArea>");
            config.append(Integer.MAX_VALUE);
            config.append("</maxArea>\n");
            config.append("<filterByCircularity>1</filterByCircularity>\n");
            config.append("<minCircularity>");
            config.append(circularity[0]);
            config.append("</minCircularity>\n");
            config.append("<maxCircularity>");
            config.append(circularity[1]);
            config.append("</maxCircularity>\n");
            config.append("<filterByInertia>1</filterByInertia>\n");
            config.append("<minInertiaRatio>0.1</minInertiaRatio>\n");
            config.append("<maxInertiaRatio>" + Integer.MAX_VALUE + "</maxInertiaRatio>\n");
            config.append("<filterByConvexity>1</filterByConvexity>\n");
            config.append("<minConvexity>0.95</minConvexity>\n");
            config.append("<maxConvexity>" + Integer.MAX_VALUE + "</maxConvexity>\n");
            config.append("</opencv_storage>\n");
            FileWriter writer;
            writer = new FileWriter(tempFile, false);
            writer.write(config.toString());
            writer.close();
            blobDet.read(tempFile.getPath());
        } catch (IOException e) {
            e.printStackTrace();
        }

        blobDet.detect(input, blobList);
    }

    /**
     * An indication of which type of filter to use for a blur.
     * Choices are BOX, GAUSSIAN, MEDIAN, and BILATERAL
     */
    enum BlurType{
        BOX("Box Blur"), GAUSSIAN("Gaussian Blur"), MEDIAN("Median Filter"),
        BILATERAL("Bilateral Filter");

        private final String label;

        BlurType(String label) {
            this.label = label;
        }

        public static BlurType get(String type) {
            if (BILATERAL.label.equals(type)) {
                return BILATERAL;
            }
            else if (GAUSSIAN.label.equals(type)) {
                return GAUSSIAN;
            }
            else if (MEDIAN.label.equals(type)) {
                return MEDIAN;
            }
            else {
                return BOX;
            }
        }

        @Override
        public String toString() {
            return this.label;
        }
    }

    /**
     * Softens an image using one of several filters.
     * @param input The image on which to perform the blur.
     * @param type The blurType to perform.
     * @param doubleRadius The radius for the blur.
     * @param output The image in which to store the output.
     */
    private static void blur(Mat input, BlurType type, double doubleRadius,
                      Mat output) {
        int radius = (int)(doubleRadius + 0.5);
        int kernelSize;
        switch(type){
            case BOX:
                kernelSize = 2 * radius + 1;
                Imgproc.blur(input, output, new Size(kernelSize, kernelSize));
                break;
            case GAUSSIAN:
                kernelSize = 6 * radius + 1;
                Imgproc.GaussianBlur(input,output, new Size(kernelSize, kernelSize), radius);
                break;
            case MEDIAN:
                kernelSize = 2 * radius + 1;
                Imgproc.medianBlur(input, output, kernelSize);
                break;
            case BILATERAL:
                Imgproc.bilateralFilter(input, output, -1, radius, radius);
                break;
        }
    }
    public static void search( double[][][] colorSpace, int files, int csInt ){
        int redRight = 0, yellowRight = 0, greenRight = 0;
        String fileName;
        for( int i = 1; i <= files; i++ ){
            int red = 0;
            double redMax = 0;
            int yellow = 0;
            double yellowMax = 0;
            int green = 0;
            double greenMax = 0;
            fileName = "sample (" + i + ").png";
            try {
                BufferedImage bi = ImageIO.read(new File(fileName));
            }catch( Exception e ){

            }
            Mat in = Imgcodecs.imread(fileName,Imgcodecs.IMREAD_COLOR);
            Mat outFinal = in.clone();
            switch( csInt ) {
                case 0:
                    Imgproc.cvtColor(in, in, Imgproc.COLOR_BGR2RGB);
                    break;
                case 1:
                    Imgproc.cvtColor(in, in, Imgproc.COLOR_BGR2HSV);
                    break;
                case 2:
                    Imgproc.cvtColor(in, in, Imgproc.COLOR_BGR2YCrCb);
                    break;
            }
            //Imgproc.cvtColor( in, in, Imgproc.COLOR_BGR2RGB);
            //displayImage(bi);
            for( int color = 0; color <= 2; color++ ) {

                double[][] colorSet = colorSpace[color];
                MatOfKeyPoint out = new MatOfKeyPoint();
                Mat rgbThresholdOutput = new Mat();
                Mat cvDilateOutput = new Mat();
                Mat blurOutput = new Mat();


                double[] a = colorSet[0];
                double[] b = colorSet[1];
                double[] c = colorSet[2];

                threshold(in, a, b, c, rgbThresholdOutput);

                // Step CV_dilate0:
                Mat cvDilateSrc = rgbThresholdOutput;
                Mat cvDilateKernel = new Mat();
                org.opencv.core.Point cvDilateAnchor = new Point(-1, -1);
                double cvDilateIterations = 4.0;
                int cvDilateBordertype = Core.BORDER_CONSTANT;
                Scalar cvDilateBordervalue = new Scalar(-1);

                cvDilate(cvDilateSrc, cvDilateKernel, cvDilateAnchor, cvDilateIterations, cvDilateBordertype, cvDilateBordervalue, cvDilateOutput);

                Mat blurInput = cvDilateOutput;
                BlurType blurType = BlurType.get("Median Filter");
                double blurRadius = 10;

                blur(blurInput, blurType, blurRadius, blurOutput);

                // Step Find_Blobs0:
                Mat findBlobsInput = blurOutput;
                double findBlobsMinArea = 30;
                double[] findBlobsCircularity = {0.50, 1.0};
                findBlobs(findBlobsInput, findBlobsMinArea, findBlobsCircularity, false, out);
                KeyPoint[] keypts = out.toArray();
                if( color == 0 ){
                    red = keypts.length;
                    double max = 0;
                    for( KeyPoint k : keypts ){
                        if( k.size > max ){
                            max = k.size;
                        }
                    }
                    redMax = max;
                }else if( color == 1 ){
                    yellow = keypts.length;
                    double max = 0;
                    for( KeyPoint k : keypts ){
                        if( k.size > max ){
                            max = k.size;
                        }
                    }
                    yellowMax = max;
                }else{
                    green = keypts.length;
                    double max = 0;
                    for( KeyPoint k : keypts ){
                        if( k.size > max ){
                            max = k.size;
                        }
                    }
                    greenMax = max;
                }
                Features2d.drawKeypoints(outFinal, out, outFinal, new Scalar(0, 0, 255), Features2d.DrawMatchesFlags_DRAW_RICH_KEYPOINTS);
                if( i == 3 ){
                    displayImage( Mat2BufferedImage( in ) );
                    displayImage( Mat2BufferedImage( rgbThresholdOutput ));
                    displayImage( Mat2BufferedImage( cvDilateOutput ));
                    displayImage( Mat2BufferedImage( blurOutput ) );
                    displayImage( Mat2BufferedImage( outFinal ) );
                }
                //displayImage( Mat2BufferedImage(blurOutput));
            }
            String status = "N/A";
            if( redMax > yellowMax && redMax > greenMax ){
                status = "Red";
            }else if( yellowMax > redMax && yellowMax > greenMax ){
                status = "Yellow";
            }else if( greenMax > yellowMax && greenMax > redMax ){
                status = "Green";
            }

            //displayImage( Mat2BufferedImage( outFinal ) );
           // writer.println("File: " + fileName);
           // writer.println("Number of Red Blobs: " + red);
           // writer.println("RedMax: " + redMax );
           // writer.println("Number of Yellow Blobs: " + yellow);
            //writer.println("YellowMax: " + yellowMax );
            //writer.println("Number of Green Blobs: " + green);
            //writer.println("GreenMax: " + greenMax );

            String real = key.get( i-1 );
            if( real.equals( status ) ){
                if( real.equals( "Red" ) ){
                    redRight++;
                }else if( real.equals( "Yellow" ) ){
                    yellowRight++;
                }else if( real.equals( "Green") ){
                    greenRight++;
                }
            }
            writer.println("Actual Status:" + key.get(i-1) );
            writer.println("Predicted Status: " + status);
            writer.println("Correct?" + real.equals( status ) );
            writer.println("");

        }
        writer.println( "Resuts:" );
        writer.println( "# of Red Photos Correctly Guessed Red: " + redRight );
        writer.println( "# of Yellow Photos Correctly Guessed Yellow: " + yellowRight );
        writer.println( "# of Green Photos Correctly Guessed Green: " + greenRight );
    }
}
