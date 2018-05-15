package main;

/**
 * Created by Isabel on 3/28/18.
 */
public class Student {

    int studentId;
    String studentName;

    public void setName (String name) {
        studentName = name;
    }
    public String getName(int studentId){
        return studentName;
    }

    public void setId( int Id ) {
        studentId =  Id;
    }

    public int getId (String studentName) {
        //System.out.println(studentId);
        return studentId;
    }



}
