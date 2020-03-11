import React, {Component} from 'react';
import {connect} from 'react-redux';
import {fetchForums} from '../../actions';
import ForumList from '../../components/forumlist';
import { Document, Page, pdfjs } from "react-pdf";
import resume from './resume.pdf';




 pdfjs.GlobalWorkerOptions.workerSrc = `//cdnjs.cloudflare.com/ajax/libs/pdf.js/${pdfjs.version}/pdf.worker.js`;

class HomeContainer extends Component {
	render() {
	    return (
	      <Document
		file={resume}
		onLoadSuccess={this.onDocumentLoadSuccess}
	      >
		<Page pageNumber={1} height={1500} />
	      </Document>
	    );
	  }
}

export default connect()(HomeContainer);
