#include <GL/glew.h>
#include <GL/glut.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define my_PI 3.141592

static char* vsSource = "#version 130 \n\
	in vec4 aPosition; \n\
	in vec4 aColor; \n\
	// flat out vec4 vColor; \n\
   	out vec4 vColor; \n\
	uniform mat4 uscale; \n\
	uniform mat4 urotate;  \n\
	uniform mat4 utranslate; \n\
	uniform mat4 um_view; \n\
	uniform mat4 um_persp; \n\
	void main(void) { \n\
  		gl_Position = um_persp*um_view*utranslate*urotate*uscale*aPosition; \n\
  	vColor = aColor; \n\
}";

static char* fsSource = "#version 130 \n\
	// flat in vec4 vColor; \n\
  	in vec4 vColor; \n\
	void main(void) { \n\
  		gl_FragColor = vColor; \n\
	}";

GLuint vs = 0;
GLuint fs = 0;
GLuint prog = 0;

char buf[1024];
int DRAW_MODE = 0, DRAW_LINE = 0, DRAW_ROTATE = 0;
float s=0.0, t = 0.0;

/*
int num_vertices = 4, num_faces = 4;

GLfloat vertices[] = { // at center
	0.0, 0.5, 0.0, 1.0, // v0
	-0.5, -0.5, 0.3, 1.0, // v1
	0.5, -0.5, 0.3, 1.0, // v2
	0.0, -0.5, -0.5, 1.0, // v3
};


GLfloat colors[] = {
	1.0, 0.0, 0.0, 1.0,  // v0 color
	0.0, 1.0, 0.0, 1.0,  // v1 color
	0.0, 0.0, 1.0, 1.0,  // v2 color
	1.0, 0.0, 1.0, 1.0,  // v3 color
};

GLushort indices[] = {
	0, 1, 2,  // red
	1, 0, 3,  // green 
	2, 3, 0,  // blue
	3, 2, 1,  // purple 
};
*/

int num_vertices = 8, num_faces = 12;
/*
GLfloat vertices[] = {
	-0.2, -0.2, -0.2, 1.0, // 0
	-0.2, -0.2, +0.2, 1.0, // 1
	-0.2, +0.2, -0.2, 1.0, // 2
	-0.2, +0.2, +0.2, 1.0, // 3
	+0.2, -0.2, -0.2, 1.0, // 4
	+0.2, -0.2, +0.2, 1.0, // 5
	+0.2, +0.2, -0.2, 1.0, // 6
	+0.2, +0.2, +0.2, 1.0, // 7
};
*/
GLfloat vertices[] = {
	-0.5, -0.5, -0.5, 1.0, // 0
	-0.5, -0.5, +0.5, 1.0, // 1
	-0.5, +0.5, -0.5, 1.0, // 2
	-0.5, +0.5, +0.5, 1.0, // 3
	+0.5, -0.5, -0.5, 1.0, // 4
	+0.5, -0.5, +0.5, 1.0, // 5
	+0.5, +0.5, -0.5, 1.0, // 6
	+0.5, +0.5, +0.5, 1.0, // 7
};

GLfloat colors[] = {
	1.0, 0.0, 0.0, 1.0,
	0.0, 1.0, 0.0, 1.0,
	0.0, 0.0, 1.0, 1.0,
	1.0, 1.0, 0.0, 1.0,
	0.0, 1.0, 1.0, 1.0,
	1.0, 0.0, 1.0, 1.0,
	1.0, 0.5, 0.2, 1.0,
	0.2, 1.0, 1.0, 1.0,
};

GLushort indices[] = { // 36 points, 12 triangles
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
	GLuint status;
	
	printf("***** Your student number and name *****\n");
	vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vsSource, NULL);
	glCompileShader(vs);  
	glGetShaderiv(vs, GL_COMPILE_STATUS, &status);
	printf("vs compile status = %s\n", (status == GL_TRUE) ? "true" : "false");
	glGetShaderInfoLog(vs, sizeof(buf), NULL, buf);
	printf("vs log = [%s]\n", buf);
	 
	fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fsSource, NULL);
	glCompileShader(fs);  
	glGetShaderiv(fs, GL_COMPILE_STATUS, &status);
	printf("fs compile status = %s\n", (status == GL_TRUE) ? "true" : "false");
	glGetShaderInfoLog(fs, sizeof(buf), NULL, buf);
	printf("fs log = [%s]\n", buf);
	 
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
	printf("program validate status = %s\n", (status == GL_TRUE) ? "true" : "false");
	glGetProgramInfoLog(prog, sizeof(buf), NULL, buf);
	printf("validate log = [%s]\n", buf);
	glUseProgram(prog);  

	GLuint loc;
	GLuint vbo[1];
	// using vertex buffer object
	glGenBuffers(1, vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo[0]);
	glBufferData(GL_ARRAY_BUFFER, 2 * num_vertices * 4 * sizeof(GLfloat), NULL, GL_STATIC_DRAW);
	glBufferSubData(GL_ARRAY_BUFFER, 0, num_vertices * 4 * sizeof(GLfloat), vertices);
	glBufferSubData(GL_ARRAY_BUFFER, num_vertices * 4 * sizeof(GLfloat), num_vertices * 4 * sizeof(GLfloat),
		colors);

	loc = glGetAttribLocation(prog, "aPosition");
	glEnableVertexAttribArray(loc);
	glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)0);

	loc = glGetAttribLocation(prog, "aColor");
	glEnableVertexAttribArray(loc);
	glVertexAttribPointer(loc, 4, GL_FLOAT, GL_FALSE, 0, (GLvoid *)(num_vertices * 4 * sizeof(GLfloat)));

	glProvokingVertex(GL_FIRST_VERTEX_CONVENTION);
	glEnable(GL_DEPTH_TEST);


}

void mykeyboard(unsigned char key, int x, int y) {
	switch (key) {
	case 27: // ESCAPE
		exit(0);
		break;
	}
}
void mymenu(int id)
{
	if (id == -1) {
		DRAW_ROTATE = (DRAW_ROTATE + 1) % 2;
	}
	else if (id == 0)
		DRAW_MODE = 0;
	else if (id == 1)
		DRAW_MODE = 1;
	else if (id == 2)
		DRAW_MODE = 2;
	else if (id == 3)
		DRAW_LINE = 1;
	else if (id == 4)
		DRAW_LINE = 0;
	else if (id == 5) {
		DRAW_ROTATE = 0;
		DRAW_LINE = 0;
		t = 0.0;
	}

	if (DRAW_LINE == 1)
		glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
	else
		glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

	glutPostRedisplay();
}


void myidle(void) {

	if (DRAW_ROTATE == 1) {

		t += 0.003f;
		s += 0.003f;
	}
	// redisplay 
	glutPostRedisplay();
}

GLfloat m[16], m_view[16], m_ortho[16], m_persp[16];

void vec_minus(float *v3, float *v1, float *v2)
{   // v3 = v1 - v2
	for (int i = 0; i < 4; i++)
		v3[i] = v1[i] - v2[i];
}
float vec_dot_prod(float *v1, float *v2)
{   // return the dot product of v1 and v2
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

float vec_len(float *v)
{
	// return the length of v
	return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
}

void vec_normalize(float *nv, float *v)
{   // nv = normalized v
	float len;
	len = vec_len(v);
	nv[0] = v[0] / len;
	nv[1] = v[1] / len;
	nv[2] = v[2] / len;
	nv[3] = v[3] / len;
}

void vec_cross_prod(float *v3, float *v1, float *v2)
{
	// v3 = v1 x v2
	v3[0] = v1[1] * v2[2] - v1[2] * v2[1];
	v3[1] = - v1[0] * v2[2] + v1[2] * v2[0];
	v3[2] = v1[0] * v2[1] - v1[1] * v2[0];
}

void vec_assign(float* v2, float *v1)
{
	for (int i = 0; i < 4; i++)
		v2[i] = v1[i];
}

void print_mat(float *m)
{
	for (int i = 0; i < 4; i++) {
		printf("%f  %f  %f   %f\n", m[i], m[i + 4], m[i + 8], m[i + 12]);
	}
}

void LookAt(float *m_view, float *eye, float *at, float *up)
{// construct m_view by using eye, at, up
	float u[4], v[4], n[4], p[4];
	vec_assign(p, eye);
	vec_minus(n, at, eye);
	vec_normalize(n, n);
	
	for (int i = 0; i < 4; i++)
		v[i] = -vec_dot_prod(up, n)*n[i] + up[i];
	vec_normalize(v, v);
	
	vec_cross_prod(u, n, v);
	vec_normalize(u, u);

	m_view[0] = u[0];    m_view[4] = u[1];    m_view[8] = u[2];    m_view[12] = -vec_dot_prod(p, u);
	m_view[1] = v[0];    m_view[5] = v[1];    m_view[9] = v[2];    m_view[13] = -vec_dot_prod(p, v);
	m_view[2] = n[0];    m_view[6] = n[1];    m_view[10] = n[2];   m_view[14] = -vec_dot_prod(p, n);
	m_view[3] = 0.0;     m_view[7] = 0.0;     m_view[11] = 0.0;    m_view[15] = 1.0;

//	printf("m_view:\n");
//	print_mat(m_view);
}
void OrthoProj (float* m_ortho, GLfloat xmin, GLfloat xmax, GLfloat ymin, GLfloat ymax, GLfloat zmin, GLfloat zmax) {
	GLfloat vec[4], uec[4], wec[4];
	int i;

	// orthogonal projection matrix
	m_ortho[0] = 2.0 / (xmax - xmin);	m_ortho[4] = 0.0;					m_ortho[8] = 0.0;					m_ortho[12] = -(xmax + xmin) / (xmax - xmin);
	m_ortho[1] = 0.0;					m_ortho[5] = 2.0 / (ymax - ymin);	m_ortho[9] = 0.0;					m_ortho[13] = -(ymax + ymin) / (ymax - ymin);
	m_ortho[2] = 0.0;					m_ortho[6] = 0.0; 					m_ortho[10] = 2.0 / (zmax - zmin);	m_ortho[14] = -(zmax + zmin) / (zmax - zmin);
	m_ortho[3] = 0.0;					m_ortho[7] = 0.0;					m_ortho[11] = 0.0;					m_ortho[15] = 1.0;


//	printf("m_ortho:\n");
//	print_mat(m_ortho);
}
void PerspProj(float* m_persp, GLfloat xmin, GLfloat xmax, GLfloat ymin, GLfloat ymax, GLfloat zmin, GLfloat zmax)
{
	m_persp[0] = 2 * zmin / (xmax - xmin);	m_persp[4] = 0;							m_persp[8] = -(xmax + xmin) / (xmax - xmin);	m_persp[12] = 0;
	m_persp[1] = 0;							m_persp[5] = 2 * zmin / (ymax - ymin);	m_persp[9] = -(ymax + ymin) / (ymax - ymin);	m_persp[13] = 0;
	m_persp[2] = 0;							m_persp[6] = 0;							m_persp[10] = (zmax + zmin) / (zmax - zmin);	m_persp[14] = -2 * zmax * zmin / (zmax - zmin);
	m_persp[3] = 0;							m_persp[7] = 0;							m_persp[11] = 1;								m_persp[15] = 0;
}
void myFrustum(float *m, GLfloat xm, GLfloat xM, GLfloat ym, GLfloat yM, GLfloat zm, GLfloat zM)
{
	if (xm < xM && ym < yM && zm > 0 && zm < zM) {
		GLfloat xlen = xM - xm;
		GLfloat ylen = yM - ym;
		GLfloat zlen = zM - zm;
		m[0] = 2*zm/xlen;m[4] = 0.0;       m[8] = -(xM+xm)/xlen;    m[12] = 0.0;
		m[1] = 0.0;      m[5] = 2*zm/ylen; m[9] = -(yM+ym)/ylen;    m[13] = 0.0;
		m[2] = 0.0;      m[6] = 0.0;       m[10] = (zM+zm)/zlen;	m[14] = -2*zM*zm/zlen;
		m[3] = 0.0;      m[7] = 0.0;       m[11] = 1.0;				m[15] = 0.0;
	}
	else {
		m[0] = 1.0;		 m[4] = 0.0;       m[8] = 0.0;			    m[12] = 0.0;
		m[1] = 0.0;      m[5] = 1.0;	   m[9] = 0.0;    			m[13] = 0.0;
		m[2] = 0.0;      m[6] = 0.0;       m[10] = 1.0;				m[14] = 0.0;
		m[3] = 0.0;      m[7] = 0.0;       m[11] = 0.0;				m[15] = 1.0;
	}
}
void mat_translate(float *m, float* t_v)
{
	m[0] = 1.0;      m[4] = 0.0;     m[8] = 0.0;      m[12] = t_v[0];
	m[1] = 0.0;      m[5] = 1.0;     m[9] = 0.0;      m[13] = t_v[1];
	m[2] = 0.0;      m[6] = 0.0;     m[10] = 1.0;     m[14] = t_v[2];
	m[3] = 0.0;      m[7] = 0.0;     m[11] = 0.0;     m[15] = 1.0;
}
void mat_scale(float *m, float *s_v)
{
	m[0] = s_v[0];   m[4] = 0.0;     m[8] = 0.0;      m[12] = 0.0;
	m[1] = 0.0;      m[5] = s_v[1];  m[9] = 0.0;      m[13] = 0.0;
	m[2] = 0.0;      m[6] = 0.0;     m[10] = s_v[2];  m[14] = 0.0;
	m[3] = 0.0;      m[7] = 0.0;     m[11] = 0.0;     m[15] = 1.0;
}
void mat_rotate_x(float *m, float t)
{
	m[0] = 1.0;		m[4] = 0.0;		m[8] = 0.0;			m[12] = 0.0;
	m[1] = 0.0;		m[5] = cos(t);  m[9] = -sin(t);		m[13] = 0.0;
	m[2] = 0.0;		m[6] = sin(t);  m[10] = cos(t);		m[14] = 0.0;
	m[3] = 0.0;		m[7] = 0.0;     m[11] = 0.0;		m[15] = 1.0;
}

void mat_rotate_y(float *m, float t)
{
	m[0] = cos(t);   m[4] = 0.0;     m[8] = sin(t);   m[12] = 0.0;
	m[1] = 0.0;      m[5] = 1.0;     m[9] = 0.0;      m[13] = 0.0;
	m[2] = -sin(t);  m[6] = 0.0;     m[10] = cos(t);  m[14] = 0.0;
	m[3] = 0.0;      m[7] = 0.0;     m[11] = 0.0;     m[15] = 1.0;
}

void mat_rotate_z(float *m, float t)
{
	m[0] = cos(t); m[4] = -sin(t); m[8] = 0.0;  m[12] = 0.0;
	m[1] = sin(t); m[5] = cos(t);  m[9] = 0.0;  m[13] = 0.0;
	m[2] = 0.0;    m[6] = 0.0;  m[10] = 1.0; m[14] = 0.0;
	m[3] = 0.0;    m[7] = 0.0;     m[11] = 0.0; m[15] = 1.0;
}

void mydisplay(void) {
	GLuint loc;
	glClearColor(1.0f, 1.0f, 1.0f, 1.0f); // gray
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	float v_scale[3] = { 1.0, 1.0, 1.0 };
	float v_trans[3] = { 0.0, 0.0, 0.0 };

	mat_scale(m, v_scale);
	loc = glGetUniformLocation(prog, "uscale");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m);

	if (DRAW_MODE == 0) 
		mat_rotate_x(m, t);
	else if (DRAW_MODE == 1) 
		mat_rotate_y(m, t);
	else if (DRAW_MODE == 2) 
		mat_rotate_z(m, t);

	loc = glGetUniformLocation(prog, "urotate");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m);
	
	mat_translate(m, v_trans);
	loc = glGetUniformLocation(prog, "utranslate");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m);

	float eye[4] = { 1.0, 1.0, 2.0, 1.0 },
		  at[4] = { 0.0, 0.0, 0.0, 1.0 },
		  up[4] = { 0.0, 1.0, 0.0, 0.0 };
/*
	float eye[4] = { 0.1, 0.1, 0.1, 1.0 },
		at[4] = { 0.0, 0.0, 0.0, 1.0 },
		up[4] = { 0.0, 1.0, 0.0, 0.0};
*/

	LookAt(m_view, eye, at, up);
	loc = glGetUniformLocation(prog, "um_view");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m_view);

	// Default state
    // float xmin = -1.0, xmax = 1.0, ymin = -1.0, ymax = 1.0, zmin = -1.0, zmax = 1.0;
/*   
	// Example 1
	float xmin = -2.5, xmax = +2.5, ymin = -1.2, ymax = 1.2, zmin = -1.0, zmax = 1.0;
	OrthoProj(m_ortho, xmin, xmax, ymin, ymax, zmin, zmax);
	loc = glGetUniformLocation(prog, "um_ortho");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m_ortho);
*/
	
	
// Example 1	
//	float xmin = -0.5, xmax = +0.5, ymin = -0.5, ymax = +0.5, zmin = 1.0, zmax = +4.0;  

// Example 2	
//	float xmin = -0.7, xmax = +0.5, ymin = -0.5, ymax = +0.5, zmin = 1.0, zmax = +4.0;

// Example 3	
	float xmin = -0.7, xmax = +0.5, ymin = -0.5, ymax = +0.5, zmin = 1.8, zmax = +4.0;


	myFrustum(m_persp, xmin, xmax, ymin, ymax, zmin, zmax);
	loc = glGetUniformLocation(prog, "um_persp");
	glUniformMatrix4fv(loc, 1, GL_FALSE, m_persp);
	   	 
	glDrawElements(GL_TRIANGLES, num_faces * 3, GL_UNSIGNED_SHORT, indices);
	glFlush();

	glutSwapBuffers();
}


int main(int argc, char* argv[]) {
	glutInit(&argc, argv);
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGB | GLUT_DEPTH);
	glutInitWindowSize(500, 500);
	glutInitWindowPosition(0, 0);
	glutCreateWindow("*** Your Student Number and Name ***");
	glutDisplayFunc(mydisplay);
	glutIdleFunc(myidle);
	glutKeyboardFunc(mykeyboard);
	glutCreateMenu(mymenu);

	glutAddMenuEntry("-----", 100);
	glutAddMenuEntry("Rotate", -1);
	glutAddMenuEntry("x-axis", 0);
	glutAddMenuEntry("y-axis", 1);
	glutAddMenuEntry("z-axis", 2);
	glutAddMenuEntry("-----", 100);
	glutAddMenuEntry("Draw by line", 3);
	glutAddMenuEntry("Draw by Polygon", 4);
	glutAddMenuEntry("-----", 100);
	glutAddMenuEntry("Initialize", 5);


	glutAttachMenu(GLUT_RIGHT_BUTTON);
	glewInit();
	myinit();
	glutMainLoop();
	return 0;
}
