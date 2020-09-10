name := "PPS-18-deeplearning"

version := "0.1"

// The native backend for nd4j.
libraryDependencies += "org.nd4j" % "nd4j-native-platform" % "0.8.0"

// The ThoughtWorks Each library, which provides the `monadic`/`each` syntax.
libraryDependencies += "com.thoughtworks.each" %% "each" % "latest.release"
addCompilerPlugin("org.scalamacros" % "paradise" % "2.1.0" cross CrossVersion.full)

// All DeepLearning.scala built-in plugins.
libraryDependencies += "com.thoughtworks.deeplearning" %% "plugins-builtins" % "latest.release"
addCompilerPlugin("com.thoughtworks.import" %% "import" % "latest.release")

// ScalaTest.
libraryDependencies += "org.scalatest" %% "scalatest" % "3.0.8" % Test

// ScalaFX.
libraryDependencies += "org.scalafx" %% "scalafx" % "14-R19"

scalaVersion := "2.11.11"
