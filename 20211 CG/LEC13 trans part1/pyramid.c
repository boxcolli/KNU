#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define VNUM 5

static char* vsSource = "#version 120 \n\
    attribute vec4 aPosition; \n\
    attribute vec4 aColor; \n\
    varying vec4 vColor; \n\
    uniform mat4 utranslate; \n\
    void main(void) { \n\
        gl_Position = utranslate * aPosition; \n\
        vColor = aColor; \n\
    }";
static char* fsSource = "#version 120 \n\
    varying vec4 vColor; \n\
    void main(void) { \n\
        gl_FragColor = vColor; \n\
    }";
GLuint vs = 0;
GLuint fs = 0;
GLuint prog = 0;
char buf[1024];
float t = -1.0f;
GLfloat vertices[] = {
    +0.0, +0.4, +0.0, 1.0,  // 0
    +0.2, +0.0, +0.2, 1.0,  // 1
    +0.2, +0.0, -0.2, 1.0,  // 2
    -0.2, +0.0, -0.2, 1.0,  // 3
    -0.2, +0.0, +0.2, 1.0,  // 4
};
GLfloat colors[] = {
    194.0/255.0, 249.0/255.0, 187.0/255.0, 1.0,
    154.0/255.0, 209.0/255.0, 212.0/255.0, 1.0,
    204.0/255.0,  51.0/255.0,  99.0/255.0, 1.0,
     98.0/255.0, 195.0/255.0, 112.0/255.0, 1.0,
     32.0/255.0,   6.0/255.0,  59.0/255.0, 1.0,
};
GLushort indices[] = {
    0, 1, 2,
    0, 2, 3,
    0, 3, 4,
    0, 4, 1,    // side face
    1, 2, 3,
    3, 4, 1,    // bottom face
};
void myinit(void) {
    GLuint stat;

    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSource, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &stat);
    printf("vs comp stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetShaderInfoLog(vs, sizeof(buf), NULL, buf);
    printf("vs comp log  = %s\n", buf);
    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSource, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &stat);
    printf("fs comp stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetShaderInfoLog(fs, sizeof(buf), NULL, buf);
    printf("fs comp log  = %s\n", buf);

    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &stat);
    printf("prog link stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("prog link log  = %s\n", buf);
    glValidateProgram(prog);
    glGetProgramiv(prog, GL_VALIDATE_STATUS, &stat);
    printf("prog vali stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("prog vali log  = %s\n", buf);
    glUseProgram(prog);

    GLuint loc;
    GLuint vbo[1];

    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 2*5*4*sizeof(GLfloat), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 5*4*sizeof(GLfloat), vertices);
    glBufferSubData(GL_ARRAY_BUFFER, 5*4*sizeof(GLfloat), 5*4*sizeof(GLfloat), colors);
    loc = glGetAttribLocation(prog, "aPosition");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*) 0);
    loc = glGetAttribLocation(prog, "aColor");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*) (5*4*sizeof(GLfloat)));
}
void mykeyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: //ESCAPE
            exit(0);
            break;
    }
}
void myidle(void) {
    t += 0.01f;
    if (t > 1)
        t = -1.0f;
    
    glutPostRedisplay();
}
GLfloat d[4];
GLfloat m[16];
void mydisplay(void) {
    GLuint loc;
    glClearColor(244.0/255.0, 216.0/255.0, 205.0/255.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    d[0] = t;
    d[1] = sin(5.0*t) / 2.0;
    d[2] = 0.0;
    d[3] = 1.0;
    m[0] = 1.0; m[4] = 0.0; m[8] = 0.0; m[12] = d[0];
    m[1] = 0.0; m[5] = 1.0; m[9] = 0.0; m[13] = d[1];
    m[2] = 0.0; m[6] = 0.0; m[10] = 1.0; m[14] = d[2];
    m[3] = 0.0; m[7] = 0.0; m[11] = 0.0; m[15] = 1.0;

    loc = glGetUniformLocation(prog, "utranslate");
    glUniformMatrix4fv(loc, 1, GL_FALSE, m);

    glDrawElements(GL_TRIANGLES, 6*3, GL_UNSIGNED_SHORT, indices);
    glFlush();
    glutSwapBuffers();
}
int main(int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("*** 2016112905");
    glutDisplayFunc(mydisplay);
    glutKeyboardFunc(mykeyboard);
    glutIdleFunc(myidle);
    glewInit();
    myinit();
    glutMainLoop();
    return 0;
}