package com.example.isabel.watchdoguserapp;

import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.view.View;

public class MainActivity extends AppCompatActivity {
    private android.widget.Button button;
    //Firebase myFirebase;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);
        android.widget.TextView myTextView = (android.widget.TextView) findViewById(R.id.textView);
        //Firebase.setAndroidContext(getApplicationContext());
        //myFirebase = new Firebase (url: "https://watchdogfirebase.firebaseio.com/");
        button = (android.widget.Button)findViewById(R.id.button);
        button .setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                openActivity2();

            }
        });

    }
    public void openActivity2(){
        android.content.Intent intent = new android.content.Intent(this, Activity2.class);
        startActivity(intent);


    }
}
