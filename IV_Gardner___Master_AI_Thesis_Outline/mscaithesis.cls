% Own custom sub-classing of book
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{mscaithesis}

\LoadClass[10pt]{book}

\newif\if@license{\@licensefalse}
\DeclareOption{license}{\@licensetrue}
\ProcessOptions\relax

\RequirePackage[paper=a4paper,
                top=2cm,
                bottom=2cm,
                left=1.5cm,
                right=1cm]{geometry}

\RequirePackage[colorlinks=true,
                citecolor=blue,
                linkcolor=black]{hyperref}
\RequirePackage[dvipsnames]{xcolor}
\RequirePackage{changepage}
\RequirePackage{etoolbox}

% Same fonts as acmart/master information studies
\RequirePackage[T1]{fontenc}
\RequirePackage[tt=false, type1=true]{libertine}
\RequirePackage[varqu]{zi4}
% \RequirePackage[libertine]{newtxmath}


% Custom chapter head
\def\thickhrule{\leavevmode \leaders \hrule height 1ex \hfill \kern \z@}
\def\position{\centering}
%% Note the difference between the commands the one is
%% make and the other one is makes
%% Ref to /usr/share/texmf-dist/tex/latex/base/book.cls
\renewcommand{\@makechapterhead}[1]{
        \vspace*{10\p@}
        {
                \parindent \z@ \position \reset@font
                \ifnum \c@secnumdepth >\m@ne
                    \if@mainmatter
                        {\color{black}\Large \scshape \@chapapp{ \thechapter} }%  % Chapter XXXX
                        \par\nobreak%
                        \vspace*{10\p@}%
                    \fi
                \fi
                \interlinepenalty\@M%
                % \thickhrule
                \par\nobreak%
                \vspace*{2\p@}%
                {\LARGE \bfseries #1\par\nobreak}%
                \par\nobreak%
                \vspace*{2\p@}%
                % \thickhrule
                \vskip 20\p@%
                \vskip 0\p@%
        }
}

%% This uses makes
\def\@makeschapterhead#1{%
        \vspace*{10\p@}%
        {
                \parindent \z@ \position \reset@font
                {\large \scshape \vphantom{\thechapter}}
                \par\nobreak
                \vspace*{10\p@}
                \interlinepenalty\@M
                % \thickhrule
                \par\nobreak
                \vspace*{2\p@}
                {\LARGE \bfseries #1\par\nobreak}
                \par\nobreak
                \vspace*{2\p@}
                % \thickhrule
                \vskip 20\p@
        }
}

\renewcommand\listoffigures{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listfigurename}%
      \@mkboth{\listfigurename}%
              {\listfigurename}%
    \@starttoc{lof}%
    \if@restonecol\twocolumn\fi
    \addcontentsline{toc}{chapter}{List of Figures}%
}
\renewcommand\listoftables{%
    \if@twocolumn
      \@restonecoltrue\onecolumn
    \else
      \@restonecolfalse
    \fi
    \chapter*{\listtablename}%
      \@mkboth{%
          \listtablename}%
         {\listtablename}%
    \@starttoc{lot}%
    \if@restonecol\twocolumn\fi
    \addcontentsline{toc}{chapter}{List of Tables}%
}

\def\studentnumber#1{\gdef\@studentnumber{#1}}
\def\period#1{\gdef\@period{#1}}
\def\credits#1{\gdef\@credits{#1}}
\def\examiner#1{\gdef\@examiner{#1}}
\def\dailysupervisor#1{\gdef\@dailysupervisor{#1}}
\def\uvasupervisor#1{\gdef\@uvasupervisor{#1}}
\def\addsupervisor#1{\gdef\@addsupervisor{#1}}
\renewcommand{\maketitle}{
        \begin{titlepage}
                \centering

                \vspace{5cm}

                \textsc{\Large MSc Artificial Intelligence}\\[0.2cm]
                \textsc{\Large Master Thesis}\\[0.5cm]

                \rule{.8\linewidth}{0.5mm} \\[0.4cm]
                {\huge \bfseries \@title}\\[0.4cm]
                \rule{.8\linewidth}{0.5mm} \\

                \vspace{2cm}
                {\large by\par}
                {\Large \@author\par}
                {\large\@studentnumber\par}

                \vspace{1cm}

                {\large\@period\par}
                {\large\@credits\par}

                \vspace{1cm}

                \begin{minipage}[t]{.5\textwidth}
                        Supervisor: \@dailysupervisor \hfill\\
                        \ifdefempty{\@uvasupervisor}{}{University Supervisor: \@uvasupervisor \hfill\\}
                        \ifdefempty{\@addsupervisor}{}{Additional Supervisor: \@addsupervisor \hfill\\}


                        ~\hfill Examiner: \@examiner
                \end{minipage}

                \vfill
                \vspace{2cm}
                \includegraphics[width=0.5\textwidth]{media/titlepage/uvalogo_g_instituut_voor_informatica.eps}
                \vspace{1cm}
        \end{titlepage}

        \newpage
        % \restoregeometry
        \thispagestyle{empty}
        ~
        \if@license
                \vfill
                \noindent\textit{\let\\=\relax\@title}\\

                \noindent This work by \@author is licensed under \textbf{CC-BY-NC}.\\
                \noindent\url{https://creativecommons.org/licenses/by-nc/4.0}\\
                \includegraphics[width=0.2\textwidth]{media/titlepage/by-nc.png}\\

                \noindent This license requires that reusers give credit to the creator. It allows reusers to distribute, remix, adapt, and build upon the material in any medium or format, for noncommercial purposes only.
        \fi
}