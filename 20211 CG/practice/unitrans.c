#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>

#define KEY_DEFAULT 0
#define KEY_W 1
#define KEY_A 2
#define KEY_S 3
#define KEY_D 4
#define KEY_I 5
#define D_SPEED 0.0001f

static char *vsSource = "#version 120 \n\
    attribute vec4 aPosition; \n\
    attribute vec4 aColor; \n\
    varying vec4 vColor; \n\
    uniform float uxdist; \n\
    uniform float uydist; \n\
    void main(void) { \n\
        gl_Position.x = aPosition.x + uxdist; \n\
        gl_Position.y = aPosition.y + uydist; \n\
        gl_Position.zw = aPosition.zw; \n\
        vColor = aColor; \n\
    }";
static char *fsSource = "#version 120 \n\
    varying vec4 vColor; \n\
    void main(void) { \n\
        gl_FragColor = vColor; \n\
    }";
GLuint vs = 0;
GLuint fs = 0;
GLuint prog = 0;
char buf[1024];
void myinit(void) {
    GLuint stat;

    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSource, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &stat);
    printf("vs compile stat = %s\n", (stat == GL_TRUE)? "true" : "false");
    glGetShaderInfoLog(vs, sizeof(buf), NULL, buf);
    printf("vs log = [%s]\n", buf);

    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSource, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &stat);
    printf("fs compile stat = %s\n", (stat == GL_TRUE)? "true" : "false");
    glGetShaderInfoLog(fs, sizeof(buf), NULL, buf);
    printf("fs log = [%s]\n", buf);

    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &stat);
    printf("prog link stat = %s\n", (stat == GL_TRUE)? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("prog link log = [%s]\n", buf);
    glValidateProgram(prog);
    glGetProgramiv(prog, GL_VALIDATE_STATUS, &stat);
    printf("prog valid stat = %s\n", (stat == GL_TRUE)? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("prog valid log = [%s]\n", buf);
    glUseProgram(prog);
}
int pushed = KEY_DEFAULT;
void mykeyboard(unsigned char key, int x, int y) {
    printf("keyboard input: %c\n", key);
    switch (key) {
        case 27: // ESCAPE
            exit(0);
            break;
        case 'w':
        case 'W':
            pushed = KEY_W;
            break;
        case 'a':
        case 'A':
            pushed = KEY_A;
            break;
        case 's':
        case 'S':
            pushed = KEY_S;
            break;
        case 'd':
        case 'D':
            pushed = KEY_D;
            break;
        case 'i':
        case 'I':
            pushed = KEY_I;
    }
}
GLfloat vertices[] = {
    0.0, 0.0, 0.0, 1.0,
    0.3, 0.0, 0.0, 1.0,
    0.0, 0.3, 0.0, 1.0,
};
GLfloat colors[] = {
    209.0/255.0, 179.0/255.0, 196.0/255.0, 1.0,
    232.0/255.0, 194.0/255.0, 202.0/255.0, 1.0,
    179.0/255.0, 146.0/255.0, 172.0/255.0, 1.0,
};
float xdist = 0.0f;
float ydist = 0.0f;
void myidle(void) {
    switch (pushed) {        
        case KEY_W:            
            ydist += D_SPEED;
            break;
        case KEY_A:
            xdist -= D_SPEED;
            break;
        case KEY_S:
            ydist -= D_SPEED;
            break;
        case KEY_D:
            xdist += D_SPEED;
            break;
        case KEY_I:
            xdist = 0.0f;
            ydist = 0.0f;
        default:
            break;
    }

    glutPostRedisplay();
}
void mydisplay(void) {
    GLuint loc;

    glClearColor(115.0/255.0, 93.0/255.0, 120.0/255.0, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    loc = glGetAttribLocation(prog, "aPosition");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, vertices);

    loc = glGetAttribLocation(prog, "aColor");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, colors);

    loc = glGetUniformLocation(prog, "uxdist");
    glUniform1f(loc, xdist);
    loc = glGetUniformLocation(prog, "uydist");
    glUniform1f(loc, ydist);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    glFlush();
    glutSwapBuffers();
}

int main(int argc, char *argv[]) {
    glutInit(&argc, argv);
    
    glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("********************");

    glutDisplayFunc(mydisplay);
    glutIdleFunc(myidle);
    glutKeyboardFunc(mykeyboard);

    glewInit();
    myinit();
    glutMainLoop();

    return 0;
}