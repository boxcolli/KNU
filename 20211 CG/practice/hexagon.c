#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>

static const char* vsSource = "#version 120 \n\
    attribute vec4 vertex; \n\
    attribute vec4 acolor; \n\
    varying vec4 fcolor; \n\
    void main() { \n\
        gl_Position = vertex; \n\
        fcolor = acolor; \n\
    }";
static const char* fsSource = "#version 120 \n\
    varying vec4 fcolor; \n\
    void main() { \n\
        gl_FragColor = fcolor; \n\
    }";

GLuint vs = 0;
GLuint fs = 0;
GLuint prog = 0;

char buf[1024];

void myinit()
{
    GLuint status;

    printf("2016112905 kim\n");
    // vs : vertex shader
    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSource, NULL);
    glCompileShader(vs); //compile to get .OBJ
    glGetShaderiv(vs, GL_COMPILE_STATUS, &status);
    printf("vs compile status = %s\n", (status == GL_TRUE) ? "true" : "false");
    glGetShaderInfoLog(vs, sizeof(buf), NULL, buf);
    printf("vs log = [%s]\n", buf);

    // fs : fragment shader
    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSource, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &status);
    printf("fs compile status = %s\n", (status == GL_TRUE) ? "true" : "false");
    glGetShaderInfoLog(fs, sizeof(buf), NULL, buf);
    printf("fs log = [%s]\n", buf);

    // prog : program
    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    printf("program link status = %s\n", (status == GL_TRUE) ? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("link log = [%s]\n", buf);
    glValidateProgram(prog);
    glGetProgramiv(prog, GL_VALIDATE_STATUS, &status);
    printf("program validate status = %s\n", (status = GL_TRUE) ? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("validate log = [%s]\n", buf);
    glUseProgram(prog);
}

void mykeyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case 27: //ESCAPE
            exit(0);
            break;
    }
}

GLfloat vertices[] = {
    +0.0, +0.7, 0.0, 1.0,
    +0.8, +0.3, 0.0, 1.0,
    +0.8, -0.3, 0.0, 1.0,
    +0.0, -0.7, 0.0, 1.0,
    -0.8, -0.3, 0.0, 1.0,
    -0.8, +0.3, 0.0, 1.0,
};
GLfloat colors[] = {
    242.0 / 255.0, 213.0 / 255.0, 248.0 / 255.0, 1.0,
    230.0 / 255.0, 192.0 / 255.0, 233.0 / 255.0, 1.0,
    191.0 / 255.0, 171.0 / 255.0, 203.0 / 141.0, 1.0,
    141.0 / 255.0, 137.0 / 255.0, 166.0 / 255.0, 1.0,
    234.0 / 255.0, 200.0 / 255.0, 202.0 / 255.0, 1.0,
    252.0 / 255.0, 200.0 / 255.0, 194.0 / 255.0, 1.0,
};

void mydisplay(void)
{
    GLuint loc;

    GLfloat bg[3] = {0};
    for(int i = 0; i < 6; i++) {
        bg[0] += colors[4 * i + 0];
        bg[1] += colors[4 * i + 1];
        bg[2] += colors[4 * i + 2];
    }
    bg[0] /= 6;
    bg[1] /= 6;
    bg[2] /= 6;
    glClearColor(bg[0], bg[1], bg[2], 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    loc = glGetAttribLocation(prog, "vertex");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, vertices);

    loc = glGetAttribLocation(prog, "acolor");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, colors);

    glDrawArrays(GL_TRIANGLE_FAN, 0, 6);
    /*
        GLenum mode:
            type of primitives to render
            GL_POINTS,
            GL_LINES, GL_LINE_STRIP, GL_LINE_LOOP
            GL_TRIANGLES, GL_TRIANGLE_STRIP, GL_TRIANGLE_FAN
        
        GLint first, GLsizei count: set range of array
     */

    glFlush();
}

int main(int argc, char* argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("2016112905 kim");
    glutDisplayFunc(mydisplay);
    glutKeyboardFunc(mykeyboard);
    glewInit();
    myinit();
    glutMainLoop();
    return 0;
}