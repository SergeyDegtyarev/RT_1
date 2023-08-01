import java.util.Scanner;

public class Main {

    public static int getNumberOfMaxParam(int a, int b, int c) {
        if ((a>=b)&&(a>=c)) return 1;
        if ((b>=a)&&(b>=c)) return 2;
        return 3;




        // write a body here
    }

    public static void main(String[] args) {
        String a = new String("1");
        String c = new String("1");
        System.out.println(a == c);

        Scanner scanner = new Scanner(System.in);


        String b = scanner.next();
        switch (b){
            case ("gryffindor"): System.out.println("bravery"); break;
            case ("hufflepuff"): System.out.println("loyalty"); break;
            case ("slytherin"): System.out.println("cunning"); break;
            case ("ravenclaw"): System.out.println("intellect"); break;
            default: System.out.println("not a valid house"); break;
        }




    }
}