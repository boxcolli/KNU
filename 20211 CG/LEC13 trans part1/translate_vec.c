#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>

static char* vsSource = "#version 120 \n\
    attribute vec4 aPosition; \n\
    attribute vec4 aColor; \n\
    varying vec4 vColor; \n\
    uniform vec4 udvec; \n\
    void main(void) { \n\
        gl_Position = aPosition + udvec; \n\
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
float dist = 0.0f;
GLfloat vertices[] = {
    -0.2, -0.2, -0.2, 1.0,
    -0.2, -0.2, +0.2, 1.0,
    -0.2, +0.2, -0.2, 1.0,
    -0.2, +0.2, +0.2, 1.0,
    +0.2, -0.2, -0.2, 1.0,
    +0.2, -0.2, +0.2, 1.0,
    +0.2, +0.2, -0.2, 1.0,
    +0.2, +0.2, +0.2, 1.0,
};
GLfloat colors[] = {
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
    0.0, 0.0, 1.0,
    1.0, 0.0, 0.0,
    0.0, 1.0, 0.0,
};
GLushort indices[] = {
    0, 4, 6,
    6, 2, 0,
    4, 5, 7,
    7, 6, 4,
    1, 3, 7,
    7, 5, 1,
    0, 2, 3,
    3, 1, 0,
    2, 6, 7,
    7, 3, 2,
    0, 1, 5,
    5, 4, 0,
};
void myinit(void) {
    GLuint stat;

    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSource, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &stat);
    printf("vs compile stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetShaderInfoLog(vs, sizeof(buf), NULL, buf);
    printf("vs log = [%s]\n", buf);

    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSource, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &stat);
    printf("fs compile stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetShaderInfoLog(vs, sizeof(buf), NULL, buf);
    printf("fs log = [%s]\n", buf);

    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &stat);
    printf("program link stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("link log = [%s]\n", buf);
    glValidateProgram(prog);
    glGetProgramiv(prog, GL_VALIDATE_STATUS, &stat);
    printf("program validate stat = %s\n", (stat == GL_TRUE) ? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("validate log = [%s]\n", buf);
    glUseProgram(prog);

    GLuint loc;
    GLuint vbo[1];
    glGenBuffers(1, vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
    glBufferData(GL_ARRAY_BUFFER, 2*8*4*sizeof(GLfloat), NULL, GL_STATIC_DRAW);
    glBufferSubData(GL_ARRAY_BUFFER, 0, 8*4*sizeof(GLfloat), vertices);
    glBufferSubData(GL_ARRAY_BUFFER, 8*4*sizeof(GLfloat), 8*4*sizeof(GLfloat), colors);

    loc = glGetAttribLocation(prog, "aPosition");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0); // bound to vbo[0]
    loc = glGetAttribLocation(prog, "aColor");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)(8*4*sizeof(GLfloat)));
}
void mykeyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: //ESCAPE
            exit(0);
            break;
    }
}
void myidle(void) {
    dist += 0.01f;
    if (dist > 1.5)
        dist = 0.0f;
    glutPostRedisplay();
}
void mydisplay(void) {
    GLuint loc;
    GLfloat d[4] = {0.2, 0.5, 0.0, 0.0};

    glClearColor(0.7f, 0.7f, 0.7f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    d[0] *= dist;
    d[1] *= dist;
    d[2] *= dist;

    loc = glGetUniformLocation(prog, "udvec");
    glUniform4fv(loc, 1, d);

    glDrawElements(GL_TRIANGLES, 12*3, GL_UNSIGNED_SHORT, indices);
    
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