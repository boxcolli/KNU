#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>

// vertex shader code
const static char* vsSource = "#version 120 \n\
    attribute vec4 vert; \n\
    void main(void) { \n\
        gl_Position = vert; \n\
    }";
/*
    vertex attribute:
        position
        normal vector
        color
        texture coordinates
    vec4: 4D vector
    gl_Position: built-in variable, for primitive assembly or clipping
 */

// fragment shader code
const static char* fsSource = "#version 120 \n\
    void main(void) { \n\
        gl_FragColor = vec4(1.0, 0.0, 0.0, 1.0); \n\
    }";
/*
    gl_FragColor: built-in var, RGBA value
 */

GLuint vs = 0;
GLuint fs = 0;
GLuint prog = 0;

void myinit(void) {
    // vs: create empty vertex shader
    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSource, NULL); // give source
    glCompileShader(vs);    // complie to get .OBJ
    printf("vs done\n");
    // fs: fragment shader
    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSource, NULL);
    glCompileShader(fs);    // comile to get .OBJ
    printf("fs done\n");
    // prog: program
    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);    // link to get .EXE, each for v, f processor
    glUseProgram(prog);     // execute it !
    printf("prog done\n");
}

void mykeyboard(unsigned char key, int x, int y) {
    switch (key) {
        case 27: // ESCAPE
            exit(0);
            break;
    }
}

GLfloat vertices[] = {
    -0.5, -0.5, 0.0, 1.0,
    +0.5, -0.5, 0.0, 1.0,
    -0.5, +0.5, 0.0, 1.0,
    +0.5, +0.8, 0.0, 1.0
};

void mydisplay(void) {
    GLuint loc;

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    loc = glGetAttribLocation(prog, "vert"); //query to get vert location
    glEnableVertexAttribArray(loc); // enable the generic vertex attribute array specified by index
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, vertices); // define location and data format

    glDrawArrays(GL_LINE_STRIP, 0, 4);
    /*
        GLenum mode:
            type of primitives to render
            GL_POINTS,
            GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP
            GL_TRAINGLES, GL_TRINAGLE_STRIP, GL_TRIANGLE_FAN
        
        GLint first, GLsizei count: set range of array
     */

    glFlush();
    /*
        empties all of these buffers
        causing all issued commands to be executed
        as quickly as they are accepted by the actual rendering engine
     */
}

void mymouse(int button, int state, int x, int y)
{
    /*
        which button: GLUT_LEFT_BUTTON, GLUT_MIDDLE_BUTTON, GLUT_RIGHT_BUTTON
        state of that button: GLUT_UP, GLUT_DOWN
        position in window
     */
    printf("mouse button: %d, state: %d, (%d, %d)\n", button, state, x, y);
}

int main(int argc, char* argv[]) {
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("2016112905 Kim");
    glutDisplayFunc(mydisplay);
    glutKeyboardFunc(mykeyboard);

    // mouse callback
    glutMouseFunc(mymouse);

    glewInit();
    myinit();
    glutMainLoop();
    return 0;
}