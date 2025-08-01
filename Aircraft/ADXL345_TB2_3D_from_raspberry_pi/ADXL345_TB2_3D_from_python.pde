import processing.opengl.*;
import java.net.*;
import java.io.*;
import saito.objloader.*;

Socket socket;
BufferedReader reader;

String data = "";
StringList recorded_data;
float roll, pitch, rollFiltered, pitchFiltered;

int record_status = 0, play_status = 0;
int recorded_status = 0;
int start_record_time = 0, start_play_time = 0;
int current_time = 0;
int recorded_items_number = 0;
int current_item_number = 0;
float scale = 1.0;
float camera_yaw = 0;

PShape TB2;

String timestamp = nf(year(), 4) + "-" + 
                   nf(month(), 2) + "-" + 
                   nf(day(), 2) + "_" + 
                   nf(hour(), 2) + "-" + 
                   nf(minute(), 2) + "-" + 
                   nf(second(), 2);

String filename = "../../Records/log_" + timestamp + ".txt";

PrintWriter output;

void setup() {
  output = createWriter(filename);
  size(960, 640, P3D);
  recorded_data = new StringList();
  TB2 = loadShape("tb2.obj");

  try {
    socket = new Socket("192.168.0.2", 5000); // IP Raspberry Pi
    reader = new BufferedReader(new InputStreamReader(socket.getInputStream()));
    println("✅ Підключено до Raspberry Pi");
  } catch (Exception e) {
    e.printStackTrace();
  }
}

void draw() {
  lights();
  translate(width/2, height/2, 0);
  background(33);
  textSize(18);
  text("<Esc>  - Завершити виконання та вивести графіки", -450, 240);
  text("<+>, <-> - Zoom", -450, 260);

  if (keyCode == 139) {
    scale = 1.1;
    keyCode = 0;
  }
  if (keyCode == 140) {
    scale = 0.9;
    keyCode = 0;
  }

  readSocket();

  text("Roll: " + int(rollFiltered) + "     Pitch: " + int(pitchFiltered), 150, 240);
  draw_UAV();
}

void readSocket() {
  try {
    if (reader.ready()) {
      data = reader.readLine();
      output.println(data);
      String[] items = split(data, ',');
      if (items.length > 1) {
        roll = float(items[0]);
        pitch = float(items[1]);
        rollFiltered = float(items[2]);
        pitchFiltered = float(items[3]);
      }
    }
  } catch (Exception e) {
    println("⚠️ Socket read error: " + e);
  }
}

void draw_UAV() {
  rotateX(radians(-rollFiltered));
  rotateZ(radians(pitchFiltered));
  rotateY(camera_yaw);
  TB2.scale(scale);
  shape(TB2);
  scale = 1.0;
}

void exit() {
  try {
    output.close();
    // 🔁 Назва скрипта (той самий каталог, що і скетч)
    String scriptName = "../../filters_compare.py";

    ProcessBuilder pb = new ProcessBuilder("python", scriptName, filename);
    pb.directory(new File(sketchPath())); // Встановлюємо робочу директорію

    // 🔁 Перенаправлення виводу в системну консоль
    pb.inheritIO();

    Process process = pb.start();

    int exitCode = process.waitFor();
    println("✅ Скрипт завершено. Код:", exitCode);

  } catch (Exception e) {
    e.printStackTrace();
  }
  
  exit(0)
}
