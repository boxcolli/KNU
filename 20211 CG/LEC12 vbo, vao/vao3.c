#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>

static char* vsrc = "#version 120 \n\
    attribute vec4 aPosition; \n\
    attribute vec4 aColor; \n\
    varying vec4 vColor; \n\
    void main() { \n\
        gl_Position = aPosition; \n\
        vColor = aColor; \n\
    }";
static char* fsrc = "#version 120 \n\
    varying vec4 vColor; \n\
    void main() { \n\
        gl_FragColor = vColor; \n\
    }";
GLuint vs = 0;
GLuint fs = 0;
GLuint prog = 0;
char buf[1024];
GLuint vbo[3], vao[3];
GLfloat pentagon[20];
GLfloat rectangle[16];
GLfloat triangle[12];
GLfloat colors5[] = {
    77.0/255.0, 139.0/255.0, 49.0/255.0, 255.0/255.0,
    77.0/255.0, 139.0/255.0, 49.0/255.0, 255.0/255.0, 
    77.0/255.0, 139.0/255.0, 49.0/255.0, 255.0/255.0,
    77.0/255.0, 139.0/255.0, 49.0/255.0, 255.0/255.0,
    77.0/255.0, 139.0/255.0, 49.0/255.0, 255.0/255.0,
};
GLfloat colors4[] = {
    255.0/255.0, 200.0/255.0, 0.0/255.0, 255.0/255.0,
    255.0/255.0, 200.0/255.0, 0.0/255.0, 255.0/255.0,
    255.0/255.0, 200.0/255.0, 0.0/255.0, 255.0/255.0,
    255.0/255.0, 200.0/255.0, 0.0/255.0, 255.0/255.0,
};
GLfloat colors3[] = {
    255.0/255.0, 132.0/255.0, 39.0/255.0, 255.0/255.0,
    255.0/255.0, 132.0/255.0, 39.0/255.0, 255.0/255.0,
    255.0/255.0, 132.0/255.0, 39.0/255.0, 255.0/255.0,
};
void makePentagon(float x, float y, float r) {
    // P = p + v
    float v[] = {
        0.951f, 0.309f,         //cos18, sin18
        0.0f, 1.0f,     //90
        -0.951f, 0.309f,    //162
        -0.588f, -0.809f,   //234
        0.588f, -0.809f    //306        
    };
    printf("makePentagon\n");
    for (int i = 0; i < 5; i++) {
        pentagon[i*4 + 0] = x + r * v[i*2 + 0];
        pentagon[i*4 + 1] = y + r * v[i*2 + 1];
        pentagon[i*4 + 2] = 0.0f;
        pentagon[i*4 + 3] = 1.0f;
        printf("x%.2f y%.2f z%.2f a%.2f\n", pentagon[i*4 + 0], pentagon[i*4 + 1], pentagon[i*4 + 2], pentagon[i*4 + 3]);
    }
}
void makeRectangle(float x, float y, float r) {
    float v[] = {
        0.966f, 0.259f,    //15
        -0.259f, 0.966f,   //105
        -0.966f, -0.259f,   //195
        0.259f, -0.966f,    //285
    };
    for (int i = 0; i < 4; i++) {
        rectangle[i*4 + 0] = x + r * v[i*2 + 0];
        rectangle[i*4 + 1] = y + r * v[i*2 + 1];
        rectangle[i*4 + 2] = 0.0f;
        rectangle[i*4 + 3] = 1.0f;
    }
}
void makeTriangle(float x, float y, float r) {
    float v[] = {
        0.0f, 1.0f,     //90
        -0.866f, -0.5f, //210
        0.866f, -0.5f    //330
    };
    for (int i = 0; i < 3; i++) {
        triangle[i*4 + 0] = x + r * v[i*2 + 0];
        triangle[i*4 + 1] = y + r * v[i*2 + 1];
        triangle[i*4 + 2] = 0.0f;
        triangle[i*4 + 3] = 1.0f;
    }
}
void myinit() {
    GLuint stat;

    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsrc, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &stat);
    printf("vs comp stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetShaderInfoLog(vs, sizeof(buf), NULL, buf);
    printf("vs comp log = [%s]\n", buf);

    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsrc, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &stat);
    printf("fs comp stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetShaderInfoLog(fs, sizeof(buf), NULL, buf);
    printf("fs comp log = [%s]\n", buf);

    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &stat);
    printf("prog link stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("prog link log = [%s]\n", buf);
    glValidateProgram(prog);
    glGetProgramiv(prog, GL_VALIDATE_STATUS, &stat);
    printf("prog valid stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("prog valid log = [%s]\n", buf);
    glUseProgram(prog);

    makePentagon(-0.5f, +0.5f, 0.3f);
    makeRectangle(0.0f, 0.0f, 0.3f);
    makeTriangle(+0.5f, -0.5f, 0.3f);

    /* get 3 vao */
    glGenVertexArrays(3, vao);
    glGenBuffers(3, vbo);
    GLuint loc;

    /* vao[0] */
    glBindVertexArray(vao[0]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 2*5*4*sizeof(GLfloat), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 5*4*sizeof(GLfloat), pentagon);
    glBufferSubData(GL_ARRAY_BUFFER, 5*4*sizeof(GLfloat), 5*4*sizeof(GLfloat), colors5);
    loc = glGetAttribLocation(prog, "aPosition");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*) 0);
    loc = glGetAttribLocation(prog, "aColor");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(5*4*sizeof(GLfloat)));

    /* vao[1] */
    glBindVertexArray(vao[1]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[1]);
    glBufferData(GL_ARRAY_BUFFER, 2*4*4*sizeof(GLfloat), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 4*4*sizeof(GLfloat), rectangle);
    glBufferSubData(GL_ARRAY_BUFFER, 4*4*sizeof(GLfloat), 4*4*sizeof(GLfloat), colors4);
    loc = glGetAttribLocation(prog, "aPosition");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    loc = glGetAttribLocation(prog, "aColor");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(4*4*sizeof(GLfloat)));
    

    /* vao[2]  */
    glBindVertexArray(vao[2]);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[2]);
    glBufferData(GL_ARRAY_BUFFER, 2*3*4*sizeof(GLfloat), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 3*4*sizeof(GLfloat), triangle);
    glBufferSubData(GL_ARRAY_BUFFER, 3*4*sizeof(GLfloat), 3*4*sizeof(GLfloat), colors3);
    loc = glGetAttribLocation(prog, "aPosition");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    loc = glGetAttribLocation(prog, "aColor");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(3*4*sizeof(GLfloat)));
}
void mykeyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: //ESC
            exit(0);
            break;
    }
}
void mydisplay() {
   
    glClearColor(30.0/255.0, 33.0/255.0, 43.0/255.0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glBindVertexArray(vao[0]);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 5);

    glBindVertexArray(vao[1]);
    glDrawArrays(GL_TRIANGLE_FAN, 0, 4);

    glBindVertexArray(vao[2]);
    glDrawArrays(GL_TRIANGLES, 0, 3);

    glFlush();
}
int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("*** 2016112905");
    glutDisplayFunc(mydisplay);
    glutKeyboardFunc(mykeyboard);

    glewInit();
    myinit();
    glutMainLoop();

    return 0;
}