import java.awt.*;
import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.File;

//import org.opencv.*;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

import javax.imageio.ImageIO;
import javax.swing.*;

public class Main {
    static{
        //System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
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
        //Mat m = ImageIO.read(new File("C:\\Users\\josep\\Desktop\\Test.png"));
        //BufferedImage bi = Mat2BufferedImage(m);
        System.out.println("Hello World!");
        try {
            BufferedImage bi = ImageIO.read(new File("Test.png"));
            Mat in = Imgcodecs.imread("Test.png",Imgcodecs.CV_LOAD_IMAGE_COLOR);
            //displayImage(bi);
            Mat out = null;
            Imgproc.threshold(in,out, 50,200,Imgproc.THRESH_BINARY);
            displayImage(Mat2BufferedImage(out));

        }catch (Exception e){
            // bnsystem.out.println(e);
            System.out.println(e);
        }

    }

}
