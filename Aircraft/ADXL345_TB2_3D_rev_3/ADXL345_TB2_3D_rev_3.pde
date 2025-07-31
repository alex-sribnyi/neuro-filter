import processing.opengl.*;
import processing.serial.*;
import java.awt.event.KeyEvent;
import java.io.IOException;
import saito.objloader.*;


Serial myPort;

String data="";
StringList recorded_data;
float roll, pitch;
PrintWriter output;

int record_status=0, play_status=0;
int recorded_status = 0;
int start_record_time=0, start_play_time=0;
int current_time=0;
int recorded_items_number=0;
int current_item_number=0;
float scale=1.0;
float camera_yaw=0;


PShape TB2;

void setup() {
    size (960, 640, P3D);
    myPort = new Serial(this, "COM10", 115200); // Ініціалізація COM-порту
    myPort.bufferUntil('\n');
    recorded_data = new StringList();
    TB2 = loadShape("tb2.obj");  
}

void draw() {
  
  lights();
  translate(width/2, height/2, 0);
  background(33);
  textSize(18);
  text("<1>, <2>, <3>, <4> - Вибір моделі БпЛА", -450, 240);
  text("<+>, <-> - Zoom", -450, 260);
//  text("<Space> - Почати запис треку (10 секунд)", -450, 280);
//  text("<Enter> - Програти трек у зворотному напрямку", -450, 300);
//  text("keyCode:" + keyCode, -200, 235);
  
  if(keyCode==37)
  {  camera_yaw+=0.1;
     keyCode=0;
  }
  if(keyCode==39)
  {  camera_yaw-=0.1;
     keyCode=0;
  }
  if(keyCode==49)
  {  TB2 = loadShape("tb2.obj");
     keyCode=0;
  }

  
  if(keyCode==50)
  {  TB2 = loadShape("shahed.obj");
     keyCode=0;
  }
  if(keyCode==51)
  {  TB2 = loadShape("anka.obj");
     keyCode=0;
  }
  if(keyCode==52)
  {  TB2 = loadShape("ww2.obj");
     keyCode=0;
  }
  if(keyCode==139)
  {  scale=1.1;
     keyCode=0;
  }
  if(keyCode==140)
  {  scale=0.9;
     keyCode=0;
  } 
  if(keyCode==32)
  { record_status=1;
    start_record_time=millis();
    recorded_data.clear();
    recorded_items_number=0;
    current_item_number=0;
  }
  
  
  if(record_status == 1)
  {  current_time=millis();
     if(start_record_time >= current_time - 10000)
     {  text("Запис треку, секунд...",150,260);
        text((start_record_time - (current_time -10000))/1000, 330, 260);
        recorded_data.append(data);
     }
     else
     {      
        record_status = 0;
        recorded_status = 1;
        recorded_items_number=recorded_data.size();
        recorded_data.reverse();
     }
     keyCode=0;
  }
  if(keyCode==10)
  { play_status=1;
    start_play_time=millis();
  }

  
  if(play_status == 1 && keyCode != 0)
  {   
      if(current_item_number<recorded_items_number)  
      {  // розбиття записаної строки на дві частини - до та після ","
         String items[] = split(recorded_data.get(current_item_number++), ',');
         if (items.length > 1)
         {
            // Крен, Тангаж у градусах
            roll = -float(items[0]);
            pitch = -float(items[1]);
         }
    
         current_time=millis();
         if(start_play_time >= current_time - 10000)
         {  text("Відтворення треку...",150,260);
          //  text((current_time-start_play_time)/1000, -200, 235);
        
         }
         text("Roll (Крен): " + int(-roll) + "     Pitch (Тангаж): " + int(pitch), 150, 240);
         draw_UAV();
      }
      else keyCode=0;
  }
  if(keyCode==0)
  {   text("Roll (Крен): " + int(roll) + "     Pitch (Тангаж): " + int(pitch), 150, 240);
      draw_UAV();
  }
}

// Читання даних з COM-порту
void serialEvent (Serial myPort) { 
   data = myPort.readStringUntil('\n');
  // прийняття усіх даних до символу переносу строки
  if (data != null) {
    data = trim(data);
    // формування строк, розділених символом ","
    String items[] = split(data, ',');
    if (items.length > 1) {

      // отримання значень крену та тангажу
      roll = float(items[0]);
      pitch = float(items[1]);
    }
  }
}

void draw_UAV()
{
  // Обертання об'єкту
     rotateX(radians(-roll));
     rotateZ(radians(pitch));
     rotateY(camera_yaw);
     TB2.scale(scale);    
     shape(TB2);
     scale=1.0;
}
