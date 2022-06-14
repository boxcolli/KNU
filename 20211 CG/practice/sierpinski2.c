#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define SPSK_TRIAL  100
#define CO_X        0
#define CO_Y        1
#define LI_M        0
#define LI_N        1
#define RANDF_DIGIT 3 // MAX 4

const static char* vsSource = "#version 120 \n\
    attribute vec4 vert; \n\
    void main(void) { \n\
        gl_Position = vert; \n\
    }";
const static char* fsSource = "#version 120 \n\
    void main(void) {\n\
        glFragColor = vec4(1.0, 1.0, 1.0, 1.0); \n\
    }";

GLuint vs = 0;
GLuint fs = 0;
GLuint prog = 0;
GLuint triangle[3][4];
GLfloat vertices[SPSK_TRIAL];

void myinit();
void mykeyboard(unsigned char key, int x, int y);
void mydisplay();
void spsk(float *v[], int trial); // used in main
float randf(float a, float b, int digit);

int main(int argc, char* argv[])
{
    glutInit(&argc, argv);

    printf("sierpinski with %d trials\n", SPSK_TRIAL);
    /*
    //vertices = (GLfloat **) malloc((SPSK_TRIAL+3) * sizeof(GLfloat *));
    for (int i = 0; i < SPSK_TRIAL+3; i++) {
        vertices[i] = (GLfloat *) malloc(4 * sizeof(GLfloat));
    } */
    printf("spsk allocation done\n");
    triangle[0][0] = -0.5; triangle[0][1] = -0.5; triangle[0][2] = 0.0; triangle[0][3] = 1.0;
    triangle[1][0] = +0.5; triangle[1][1] = -0.5; triangle[1][2] = 0.0; triangle[1][3] = 1.0;
    triangle[2][0] = -0.5; triangle[2][1] = +0.5; triangle[2][2] = 0.0; triangle[2][3] = 1.0;
    
    srand(time(NULL));
    spsk(vertices, SPSK_TRIAL);
    //printf("v[3][0] %.1f v[3][1] %.1f\n", vertices[3][0], vertices[3][1]);

    glutInitDisplayMode(GLUT_SINGLE | GLUT_RGB);
    glutInitWindowSize(500, 500);
    glutInitWindowPosition(0, 0);
    glutCreateWindow("2016112905 KMS");

    glutDisplayFunc(mydisplay);
    glutKeyboardFunc(mykeyboard);

    glewInit();
    myinit();
    glutMainLoop();
    return 0;
}

void myinit()
{
    vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSource, NULL);
    glCompileShader(vs);

    fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSource, NULL);
    glCompileShader(fs);

    prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
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
void mydisplay()
{
    GLuint loc;
    glClearColor(0.0, 0.0, 0.0, 1.0);
    glClear(GL_COLOR_BUFFER_BIT);
    loc = glGetAttribLocation(prog, "vert");
    glEnableVertexAttribArray(loc);
    glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, vertices);

    glDrawArrays(GL_POINTS, 0, SPSK_TRIAL);
    glFlush();
}

void spsk(float *t[], float *v, int trial)
{
/* Sierpinski Gasket Algorithm */

    float line1[2], line2[2];
    // LI_M: m from (y = mx + n)
    // LI_N: n from (y = mx + n)

    float A[2], p[2], q[2];    
    // CO_X: x coordinate
    // CO_Y: y coordinate

    int i, trand;
    
/* 1. p = random point inside of the triangle */
    // ( choose a random point A on the line v0-v1 )
    line1[LI_M] = (t[0][CO_Y] - t[1][CO_Y]) / (t[0][CO_X] - t[1][CO_X]);
    line1[LI_N] = t[0][CO_Y] - line1[LI_M] * t[0][CO_X];
    printf("line1 (%6.3f, %6.3f)\n", line1[LI_M], line1[LI_N]);
    A[CO_X] = randf(t[0][CO_X], t[1][CO_X], RANDF_DIGIT);
    A[CO_Y] = line1[LI_M] * A[CO_X] + line1[LI_N];
    printf("randA (%6.3f, %6.3f)\n", A[CO_X], A[CO_Y]);

    // ( choose a random point p on the line A-v2 )
    line2[LI_M] = (A[CO_Y] - t[2][CO_Y]) / (A[CO_X] - t[2][CO_X]);
    line2[LI_N] = A[CO_Y] - line2[0] * A[CO_X];
    printf("line2 (%6.3f, %6.3f)\n", line2[LI_M], line2[LI_N]);
    p[CO_X] = randf(A[CO_X], t[2][CO_X], RANDF_DIGIT);
    p[CO_Y] = line2[LI_M] * p[CO_X] + line2[LI_N];
    printf("randp (%6.3f, %6.3f)\n", p[CO_X], p[CO_Y]);

    for (i = 0; i < trial; i++) {
/* 2. v = random vertex of triangle */        
        trand = rand() % 3;

/* 3. find q = (p+v)/2 */
        q[CO_X] = (p[CO_X] + t[trand][CO_X]) / 2;
        q[CO_Y] = (p[CO_Y] + t[trand][CO_Y]) / 2;

/* 4. write q */
        (v + 4*i)[0] = q[0];
        (v + 4*i)[1] = q[1];
        (v + 4*i)[2] = 0.0;
        (v + 4*i)[3] = 1.0;
        
/* 5. next p is q */
        p[CO_X] = q[CO_X];
        p[CO_Y] = q[CO_Y];

/* 6. repeat from 2 */
        //printf("spsk trial %d th: q(%.3f %.3f)\n", i, v[i+3][0], v[i+3][1]);
    }
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
    for(i = 0; i < digit; i++) {
        A *= 10;
        B *= 10;        
    }
    // choose rand from range(min, max)
    if (B > 0.0)
        result = (float) ( rand() % ((int)B - (int)A + 1) + (int)A );
    else {
        result = (float) ( rand() % -((int)B - (int)A + 1) + (int)A );
    }
    // make result small
    for(i = 0; i < digit; i++)
        result /= 10;
    return result;
}