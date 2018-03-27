package com.watchdog;
import java.util.Scanner;
import java.util.*;

public class Main {

    public static void main(String[] args) {
        int sushi = 15;
        int fizzremainder = sushi % 3;
        int buzzremainder = sushi % 5;
        int fizzbuzzremainder = sushi % 15;

        if (fizzbuzzremainder == 0) {
            System.out.println("fizzbuzz");
        } else if (fizzremainder == 0) {
            System.out.println("fizz");
        } else if (buzzremainder == 0) {
            System.out.println("buzz");
        } else {
            System.out.println(sushi);
        }







        ArrayList<int[]> arl =new ArrayList<int[]>();
        arl.add(1);
        arl.add(22);
        arl.add(-2);
        /*int[] anArray;
        int[] finalArray;
        finalArray = new int[20];
        anArray = new int[20];
        anArray[0] = 100;
        anArray[1] = 200;
        anArray[2] = 300;
        anArray[3] = 5;
        anArray[4] = 3;
        int finalNum = 0;
        int secondNum = 0;
        int counter =0;
        for (int i = 0; i < anArray.length - 1; i++) {
            if (finalArray[counter] < anArray[i]) {
                finalArray[counter] = anArray[i];
                counter++;
                anArray[i] = 0;
            }
            if (secondNum < anArray[i]){
                secondNum = anArray[i];
                finalArray[1] = anArray[i];
                anArray[i] = 0;
            }



        }
        System.out.println(finalArray[0] + " " + finalArray[1]);

*/
    }

    }





