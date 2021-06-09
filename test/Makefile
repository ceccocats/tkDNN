DEBUG= -g
OPENCV= -lopencv_core -lopencv_highgui -lopencv_imgproc -lopencv_video

PKG_CONF_CFLAG= 'pkg-config --cflags opencv --libs opencv'
PKGCONF_LIBS= 'pkg-config --libs opencv'

client:
	clear
	g++ -std=c++11 $(PKG_CONF_CFLAG) -o client client.cpp $(PKGCONF_LIBS)

server:
	clear
	g++ -std=c++11 $(PKG_CONF_CFLAG) -o server server.cpp $(PKGCONF_LIBS)
