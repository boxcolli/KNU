#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define my_PI 3.141592

static char* vsSource = "#version 130 \n\
    in vec4 aPosition; \n\
    in vec4 aColor; \n\
    flat out vec4 vColor; \n\
    uniform mat4 urotate; \n\
    uniform mat4 utranslate; \n\
    void main(void) { \n\
        gl_Position = aPosition; \n\
        gl_Position = urotate * gl_Position; \n\
        gl_Position = gl_Position; \n\
        vColor = aColor; \n\
    }";
static char* fsSource = "#version 130 \n\
    flat in vec4 vColor; \n\
    void main(void) { \n\
        gl_FragColor = vColor; \n\
    }";
GLuint vs = 0;
GLuint fs = 0;
GLuint pr = 0;
char buf[1024];
int DRAW_MODE = 0;
float t = 0.0f;
int num_vertices = 4, num_faces = 4;
GLfloat vertices[] = {
    0.0, 0.5, 0.0, 1.0,
    -0.5, -0.5, 0.3, 1.0,
    0.5, -0.5, 0.3, 1.0,
    0.0, -0.5, -0.5, 1.0,
};
GLfloat colors[] = {
    1.0, 0.0, 0.0, 1.0,
    0.0, 1.0, 0.0, 1.0,
    0.0, 0.0, 1.0, 1.0,
    1.0, 0.0, 1.0, 1.0,
};
GLushort indices[] = {
    0, 1, 2, //R
    1, 0, 3, //G
    2, 3, 0, //B
    3, 2, 1, //Purple
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

    GLuint loc;
    GLuint vbo[1];
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 2*num_vertices*4*sizeof(GLfloat), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, num_vertices*4*sizeof(GLfloat), vertices);
    glBufferSubData(GL_ARRAY_BUFFER, num_vertices*4*sizeof(GLfloat), num_vertices*4*sizeof(GLfloat), colors);

    loc = glGetAttribLocation(pr, "aPosition");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);
    loc = glGetAttribLocation(pr, "aColor");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid*)(num_vertices*4*sizeof(GLfloat)));

    glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);
    glEnable(GL_DEPTH_TEST);
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}
void mykeyboard(unsigned char key, int x, int y) {
    switch (key) {
    case 27:
        exit(0);
        break;
    }
}
void myidle(void) {
    t += 0.01f;

    glutPostRedisplay();
}

GLfloat m[16];

void mydisplay(void) {
    GLuint loc;
    glClearColor(0.7f, 0.7f, 0.7f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    // urotate : R(90')
    t = -my_PI/2.0;
    m[0] = cos(t);  m[4] = 0.0;     m[8] = sin(t);      m[12] = 0.0;
    m[1] = 0.0;     m[5] = 1.0;     m[9] = 0.0;         m[13] = 0.0;
    m[2] = -sin(t); m[6] = 0.0;     m[10] = cos(t);     m[14] = 0.0;
    m[3] = 0.0;     m[7] = 0.0;     m[11] = 0.0;        m[15] = 1.0;
    loc = glGetUniformLocation(pr, "urotate");
    glUniformMatrix4fv(loc, 1, GL_FALSE, m);

    // utranslate : Tx()
    m[0] = 1.0;     m[4] = 0.0;     m[8] = 0.0;         m[12] = -0.4;
    m[1] = 0.0;     m[5] = 1.0;     m[9] = 0.0;         m[13] = 0.0;
    m[2] = 0.0;     m[6] = 0.0;     m[10] = 1.0;        m[14] = 0.0;
    m[3] = 0.0;     m[7] = 0.0;     m[11] = 0.0;        m[15] = 1.0;
    loc = glGetUniformLocation(pr, "utranslate");
    glUniformMatrix4fv(loc, 1, GL_FALSE, m);

    glDrawElements(GL_TRIANGLES, num_faces*3, GL_UNSIGNED_SHORT, indices);
    glFlush();

    glutSwapBuffers();
}
int main(int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("*** YOUR STUDENT NUMBER AND NAME ***");
    glutDisplayFunc(mydisplay);
    //glutIdleFunc(myidle);
    glutKeyboardFunc(mykeyboard);
    glewInit();
    myinit();
    glutMainLoop();
    return 0;
}