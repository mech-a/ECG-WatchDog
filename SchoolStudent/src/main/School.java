package main;

/**
 * Created by Isabel on 3/28/18.
 */
public class School {
    public static void main(String[] args) {
        // Following statement would create an object studentName
        Student newStudent = new Student();
        newStudent.setName ("Gaurav");
        newStudent.getName (100);
        newStudent.setId (100);
        newStudent.getId("Gaurav");
        System.out.println (newStudent.studentName + " "+ newStudent.studentId);
    }
}
