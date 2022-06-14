#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>

#define MAX_VERTICES 5
#define RANDF_DIGIT 3

const static char *vsSource = "#version 120 \n\
    attribute vec4 vertex; \n\
    attribute vec4 acolor; \n\
    varying vec4 fcolor; \n\
    void main() { \n\
        gl_Position = vertex; \n\
        fcolor = acolor; \n\
    }";
const static char *fsSource = "#version 120 \n\
    varying vec4 fcolor; \n\
    void main() { \n\
        gl_FragColor = fcolor; \n\
        }";

GLuint vs = 0;
GLuint fs = 0;
GLuint prog = 0;

char buf[1024];

void myinit();
void mykeyboard(unsigned char key, int x, int y);

GLfloat vertices[MAX_VERTICES * 4];
GLfloat colors[MAX_VERTICES * 4];
int v_shape;

void mydisplay();
void mymenu(int id);

void vc_randshape(int shape, GLfloat *v, GLfloat *c);
float randf(float A, float B, int digit);

int main(int argc, char* argv[])
{
    glutInit(&argc, argv);
    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("3 shape redisplay with menu");
    glutDisplayFunc(mydisplay);
    glutKeyboardFunc(mykeyboard);

    int menu_id = glutCreateMenu(mymenu);
    glutAddMenuEntry("Triangle", 1);
    glutAddMenuEntry("Rectangle", 2);
    glutAddMenuEntry("Pentagon", 3);    
    glutAttachMenu(GLUT_RIGHT_BUTTON);

    glewInit();
    myinit();
    glutMainLoop();
    return 0;
}

void myinit()
{
    GLuint status;

    // vertex shader
    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSource, NULL);
    glCompileShader(vs);
    glGetShaderiv(vs, GL_COMPILE_STATUS, &status);
    printf("vs compile status = %s\n", (status == GL_TRUE) ? "true" : "false");
    glGetShaderInfoLog(vs, sizeof(buf), NULL, buf);
    printf("vs log = [%s]\n", buf);

    // fragment shader
    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSource, NULL);
    glCompileShader(fs);
    glGetShaderiv(fs, GL_COMPILE_STATUS, &status);
    printf("fs compile status = %s\n", (status == GL_TRUE)? "true" : "false");
    glGetShaderInfoLog(fs, sizeof(buf), NULL, buf);
    printf("fs log = [%s]\n", buf);

    // program    
    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    glGetProgramiv(prog, GL_LINK_STATUS, &status);
    printf("prog link status = %s\n", (status == GL_TRUE)? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("link log = [%s]\n", buf);
    glValidateProgram(prog);
    glGetProgramiv(prog, GL_VALIDATE_STATUS, &status);
    printf("prog validate status = %s\n", (status == GL_TRUE)? "true" : "false");
    glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
    printf("validate log = [%s]\n", buf);
    glUseProgram(prog);

    // srand !
    srand(time(NULL));
    v_shape = 3;
    vc_randshape(v_shape, vertices, colors);
}
void mykeyboard(unsigned char key, int x, int y)
{
    switch (key) {
        case 27:
            // ESCAPE
            exit(0);
            break;
    }
}
void mydisplay()
{
    GLuint loc;

    glClearColor(1.0, 1.0, 1.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);

    loc = glGetAttribLocation(prog, "vertex");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, vertices);

    loc = glGetAttribLocation(prog, "acolor");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, colors);

    glDrawArrays(GL_TRIANGLE_FAN, 0, v_shape);

    glFlush();
}
void mymenu(int id)
{
    switch(id) {        
        case 1:
            // Triangle
            v_shape = 3;
            break;
        case 2:
            // Rectangle
            v_shape = 4;
            break;
        case 3:
            // Pentagon
            v_shape = 5;
            break;        
    }
    vc_randshape(v_shape, vertices, colors);
    glutPostRedisplay();
}
void vc_randshape(int shape, GLfloat *v, GLfloat *c)
{
    // need vector (cos, sin) with magnitude 1
    float temp;
    // need scalar
    float a;

    for (int i = 0; i < shape; i++) {
        a = randf(0.0, 1.0, RANDF_DIGIT);
        temp = randf(-1.0, 1.0, RANDF_DIGIT); // x = randf(-1.0, 1.0)
        v[4 * i + 0] = a * temp;

        temp = sqrtf(1.0 - temp * temp); // y = sqrt(1 - x^2)
        if (rand() % 2 == 0) temp *= -1; // y randomly negative
        v[4 * i + 1] = a * temp;
        v[4 * i + 2] = 1.0;
        v[4 * i + 3] = 1.0;

        c[4 * i + 0] = randf(0.0, 1.0, RANDF_DIGIT);
        c[4 * i + 1] = randf(0.0, 1.0, RANDF_DIGIT);
        c[4 * i + 2] = randf(0.0, 1.0, RANDF_DIGIT);
        c[4 * i + 3] = 1.0;
    }    
}
void randf_direction(float *d)
{
    // choose x in range(-1, 1)
    d[0] = randf(-1.0, 1.0, RANDF_DIGIT);
    // choose y
    d[1] = sqrtf(1.0 - d[0] * d[0]);
    // just randomly make y negative
    if (rand() % 2 == 0)
        d[1] *= -1;
}
float randf(float A, float B, int digit)
{
	int i;
	float result;

	// confirm A < B
	if (A > B) {
		result = A; // using result as temp
		A = B;
		B = result;
	}
	// make floats big
	for (i = 0; i < digit; i++) {
		A *= 10;
		B *= 10;
	}
	// choose rand from range(min, max)
	if (B > 0.0)
		result = (float)(rand() % ((int)B - (int)A + 1) + (int)A);
	else
		result = (float)(rand() % -((int)B - (int)A + 1) + (int)A);

	// make result small
	for (i = 0; i < digit; i++)
		result /= 10;
	return result;
}