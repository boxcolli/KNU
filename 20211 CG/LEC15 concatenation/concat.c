#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
/*
    1. Construct the object by given arrays
    2. Compute the centroid
    3. Translate the object for the centroid to be at the origin
    4. Rotate the object by Euler angles:
        θx = 30 degree
        θy = 30 degree
        θz = 30 degree
        by using rotation matrix
        must use cos() sin() functions
    5. Draw the rotated object back to it's original position
    T(+pf) R(30') T(-pf)
*/
#define my_PI 3.141592
#define RAD(degree) degree * my_PI / 180.0
#define ROT_DEG 30

static char* vsSource = "#version 130 \n\
    attribute vec4 aPosition; \n\
    attribute vec4 aColor; \n\
    varying vec4 vColor; \n\
    \n\
    uniform mat4 urx_m4; \n\
    uniform mat4 ury_m4; \n\
    uniform mat4 urz_m4; \n\
    uniform vec3 ucent_v3; \n\
    \n\
    void main(void) { \n\
        mat4 tran_m4 = mat4(1.0); \n\
        mat4 concat_m4; \n\
        \n\
        tran_m4[3] = vec4(-ucent_v3, 1.0); \n\
        concat_m4 = tran_m4; \n\
        \n\
        concat_m4 = urx_m4 * concat_m4; \n\
        concat_m4 = ury_m4 * concat_m4; \n\
        concat_m4 = urz_m4 * concat_m4; \n\
        \n\
        tran_m4[3] = vec4(ucent_v3, 1.0); \n\
        concat_m4 = tran_m4 * concat_m4; \n\
        \n\
        gl_Position = concat_m4 * aPosition; \n\
        vColor = aColor; \n\
    }";
static char* fsSource = "#version 130 \n\
    varying vec4 vColor; \n\
    void main(void) { \n\
        gl_FragColor = vColor; \n\
    }";
GLuint vs = 0;
GLuint fs = 0;
GLuint pr = 0;
char buf[1024];
GLfloat vertices[] = {
    0.5, 0.8, 0.0, 1.0,     //0
    0.3, 0.3, +0.2, 1.0,    //1
    0.7, 0.3, +0.2, 1.0,    //2
    0.7, 0.3, -0.2, 1.0,    //3
    0.3, 0.3, -0.2, 1.0,    //4
};
GLfloat colors[] = {
    1.0, 0.0, 0.0 ,1.0,
    0.0, 1.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 1.0,
    1.0, 0.0, 1.0, 1.0,
    1.0, 1.0, 0.0, 1.0
};
GLushort indices[] = {
    0, 1, 2,
    2, 3, 0,
    4, 0, 3,
    1, 0, 4,
    2, 3, 1,
    3, 4, 1
};
void myinit(void) {
    printf("***** 2016112905 Kim *****\n");

    GLuint stat;
    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSource, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &stat);
    printf("vs comp stat = %s\n", (stat == GL_TRUE)? "true" : "false");
    glGetShaderInfoLog(vs, sizeof(buf), NULL, buf);
    printf("vs log       = %s\n", buf);
    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSource, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &stat);
    printf("fs comp stat = %s\n", (stat == GL_TRUE)? "true" : "false");
    glGetShaderInfoLog(vs, sizeof(buf), NULL, buf);
    printf("fs log       = %s\n", buf);
    pr = glCreateProgram();
    glAttachShader(pr, vs);
    glAttachShader(pr, fs);
    glLinkProgram(pr);
    glGetProgramiv(pr, GL_LINK_STATUS, &stat);
    printf("pr link stat = %s\n", (stat == GL_TRUE)? "true" : "false");
    glGetProgramInfoLog(pr, sizeof(buf), NULL, buf);
    printf("pr link log  = %s\n", buf);
    glValidateProgram(pr);
    glGetProgramiv(pr, GL_VALIDATE_STATUS, &stat);
    printf("pr vald stat = %s\n", (stat == GL_TRUE)? "true" : "false");
    glGetProgramInfoLog(pr, sizeof(buf), NULL, buf);
    printf("pr vald log  = %s\n", buf);
    glUseProgram(pr);

    GLuint vbo[1];
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 2*5*4*sizeof(GLfloat), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 5*4*sizeof(GLfloat), vertices);
    glBufferSubData(GL_ARRAY_BUFFER, 5*4*sizeof(GLfloat), 5*4*sizeof(GLfloat), colors);
    GLuint loc;
    loc = glGetAttribLocation(pr, "aPosition");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    loc = glGetAttribLocation(pr, "aColor");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(5*4*sizeof(GLfloat)));

    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
void mykeyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: //ESCAPE
            exit(0);
            break;
    }
}
void mydisplay(void) {
    glClearColor(177.0/255.0, 248.0/255.0, 242.0/255.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    GLuint loc;
    
    GLfloat centroid[3] = { 0.0 };
    int i;
    for (i = 0; i < 5; i++) {
        centroid[0] += vertices[i*4 + 0];
        centroid[1] += vertices[i*4 + 1];
        centroid[2] += vertices[i*4 + 2];
    }
    centroid[0] /= 5;
    centroid[1] /= 5;
    centroid[2] /= 5;
    loc = glGetUniformLocation(pr, "ucent_v3");
    glUniform3fv(loc, 1, centroid);

    GLfloat r = RAD(ROT_DEG);
    GLfloat rx[16];
    rx[0] = 1.0;        rx[4] = 0.0;        rx[8] = 0.0;        rx[12] = 0.0;
    rx[1] = 0.0;        rx[5] = cos(r);     rx[9] = -sin(r);    rx[13] = 0.0;
    rx[2] = 0.0;        rx[6] = sin(r);     rx[10] = cos(r);    rx[14] = 0.0;
    rx[3] = 0.0;        rx[7] = 0.0;        rx[11] = 0.0;       rx[15] = 1.0;

    GLfloat ry[16];
    ry[0] = cos(r);     ry[4] = 0.0;        ry[8] = sin(r);     ry[12] = 0.0;
    ry[1] = 0.0;        ry[5] = 1.0;        ry[9] = 0.0;        ry[13] = 0.0;
    ry[2] = -sin(r);    ry[6] = 0.0;        ry[10] = cos(r);    ry[14] = 0.0;
    ry[3] = 0.0;        ry[7] = 0.0;        ry[11] = 0.0;       ry[15] = 1.0;

    GLfloat rz[16];
    rz[0] = cos(r);     rz[4] = -sin(r);    rz[8] = 0.0;        rz[12] = 0.0;
    rz[1] = sin(r);     rz[5] = cos(r);     rz[9] = 0.0;        rz[13] = 0.0;
    rz[2] = 0.0;        rz[6] = 0.0;        rz[10] = 1.0;       rz[14] = 0.0;
    rz[3] = 0.0;        rz[7] = 0.0;        rz[11] = 0.0;       rz[15] = 1.0;  
    
    loc = glGetUniformLocation(pr, "urx_m4");
    glUniformMatrix4fv(loc, 1, GL_FALSE, rx);
    loc = glGetUniformLocation(pr, "ury_m4");
    glUniformMatrix4fv(loc, 1, GL_FALSE, ry);
    loc = glGetUniformLocation(pr, "urz_m4");
    glUniformMatrix4fv(loc, 1, GL_FALSE, rz);

    glDrawElements(GL_TRIANGLES, 6*3, GL_UNSIGNED_SHORT, indices);
    glFlush();
    glutSwapBuffers();
}
int main(int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("*** 2016112905 kim ***");
    glutDisplayFunc(mydisplay);
    glutKeyboardFunc(mykeyboard);
    glewInit();
    myinit();
    glutMainLoop();
    return 0;
}