// Import the functions you need from the SDKs you need
import { initializeApp } from "firebase/app";
import { getFirestore } from "firebase/firestore";
// TODO: Add SDKs for Firebase products that you want to use
// https://firebase.google.com/docs/web/setup#available-libraries

// Your web app's Firebase configuration
// For Firebase JS SDK v7.20.0 and later, measurementId is optional
const firebaseConfig = {
  apiKey: "AIzaSyCBr7sDrXV9-qBRbywqoTKkRWX3tIDsOF0",
  authDomain: "gym-posture-monitoring.firebaseapp.com",
  projectId: "gym-posture-monitoring",
  storageBucket: "gym-posture-monitoring.firebasestorage.app",
  messagingSenderId: "1051844221099",
  appId: "1:1051844221099:web:7801782b59865080507fee",
  measurementId: "G-DY7JZN8HSG"
};

// Initialize Firebase
const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

export { db };
